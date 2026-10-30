use parry3d_f64::math::{Mat3, Pose3, Rot3, Vec3};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods, IntoPyArray};
use parry3d_f64::shape::{
    Ball, Capsule as ParryCapsule, Compound, ConvexPolyhedron, Cuboid,
    Cylinder as ParryCylinder, SharedShape, TriMesh as ParryTriMesh,
};
use parry3d_f64::query;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Shape Types
// ============================================================================

/// A box shape with given half-extents.
#[pyclass(module = "py_parry3d._internal", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct Box {
    half_extents: [f64; 3],
}

#[pymethods]
impl Box {
    #[new]
    fn new(half_extents: [f64; 3]) -> Self {
        Box { half_extents }
    }

    fn __repr__(&self) -> String {
        format!("Box(half_extents={:?})", self.half_extents)
    }
}

impl Box {
    fn to_shared_shape(&self) -> (SharedShape, Pose3) {
        (SharedShape::new(Cuboid::new(Vec3::new(
            self.half_extents[0],
            self.half_extents[1],
            self.half_extents[2],
        ))), Pose3::IDENTITY)
    }
}

/// A sphere shape with given radius.
#[pyclass(module = "py_parry3d._internal", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct Sphere {
    radius: f64,
}

#[pymethods]
impl Sphere {
    #[new]
    #[pyo3(signature = (radius))]
    fn new(radius: f64) -> Self {
        Sphere { radius }
    }

    fn __repr__(&self) -> String {
        format!("Sphere(radius={})", self.radius)
    }
}

impl Sphere {
    fn to_shared_shape(&self) -> (SharedShape, Pose3) {
        (SharedShape::new(Ball::new(self.radius)), Pose3::IDENTITY)
    }
}

/// A capsule shape (cylinder with hemispherical caps) along Z axis.
#[pyclass(module = "py_parry3d._internal", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct Capsule {
    half_height: f64,
    radius: f64,
}

#[pymethods]
impl Capsule {
    #[new]
    fn new(half_height: f64, radius: f64) -> Self {
        Capsule { half_height, radius }
    }

    fn __repr__(&self) -> String {
        format!("Capsule(half_height={}, radius={})", self.half_height, self.radius)
    }
}

impl Capsule {
    fn to_shared_shape(&self) -> (SharedShape, Pose3) {
        // Z-axis aligned capsule
        let capsule = ParryCapsule::new_z(self.half_height, self.radius);
        (SharedShape::new(capsule), Pose3::IDENTITY)
    }
}

/// A cylinder shape along Z axis.
#[pyclass(module = "py_parry3d._internal", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct Cylinder {
    half_height: f64,
    radius: f64,
}

#[pymethods]
impl Cylinder {
    #[new]
    fn new(half_height: f64, radius: f64) -> Self {
        Cylinder { half_height, radius }
    }

    fn __repr__(&self) -> String {
        format!("Cylinder(half_height={}, radius={})", self.half_height, self.radius)
    }
}

impl Cylinder {
    /// parry3d's `Cylinder` is along the Y axis; ours is along Z. The
    /// reorientation is returned as the shape's intrinsic pose offset rather
    /// than baked in with a one-element `Compound`: a `Compound` here would
    /// nest inside the `Compound` a multi-object `CollisionGroup` builds, and
    /// parry panics on nested composite shapes (#9).
    fn to_shared_shape(&self) -> (SharedShape, Pose3) {
        // Rotate 90 degrees around X axis: Y -> Z
        let rotation = Rot3::from_axis_angle(Vec3::X, std::f64::consts::FRAC_PI_2);
        let cylinder = ParryCylinder::new(self.half_height, self.radius);
        (SharedShape::new(cylinder), Pose3::from_rotation(rotation))
    }
}

/// A triangle mesh shape.
///
/// **IMPORTANT: TriMesh is HOLLOW (surface-only)**
///
/// TriMesh only detects collisions with the mesh surface. Objects fully
/// inside the mesh will NOT be detected as colliding. This is different
/// from ConvexHull which is solid.
///
/// Use TriMesh when:
/// - You need exact mesh geometry for collision
/// - Surface contact detection is sufficient
/// - Objects being inside the mesh is acceptable
///
/// Use ConvexHull when:
/// - You need solid collision detection
/// - Simplified geometry is acceptable
#[pyclass(module = "py_parry3d._internal", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct TriMesh {
    vertices: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
}

#[pymethods]
impl TriMesh {
    #[new]
    fn new(vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>) -> PyResult<Self> {
        let verts_shape = vertices.shape();
        let faces_shape = faces.shape();

        if verts_shape.len() != 2 || verts_shape[1] != 3 {
            return Err(PyValueError::new_err("vertices must be (N, 3) array"));
        }
        if faces_shape.len() != 2 || faces_shape[1] != 3 {
            return Err(PyValueError::new_err("faces must be (M, 3) array"));
        }

        let verts: Vec<[f64; 3]> = vertices
            .as_slice()?
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        let face_indices: Vec<[u32; 3]> = faces
            .as_slice()?
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        Ok(TriMesh {
            vertices: verts,
            faces: face_indices,
        })
    }

    fn __repr__(&self) -> String {
        format!("TriMesh(vertices={}, faces={})", self.vertices.len(), self.faces.len())
    }
}

impl TriMesh {
    fn to_shared_shape(&self) -> (SharedShape, Pose3) {
        let points: Vec<Vec3> = self.vertices
            .iter()
            .map(|v| Vec3::new(v[0], v[1], v[2]))
            .collect();

        let trimesh = ParryTriMesh::new(points, self.faces.clone())
            .expect("Failed to create TriMesh");
        (SharedShape::new(trimesh), Pose3::IDENTITY)
    }
}

/// A convex hull computed from mesh vertices.
///
/// **IMPORTANT: ConvexHull is SOLID**
///
/// ConvexHull detects collisions with objects both touching the surface
/// AND fully inside the hull. This is different from TriMesh which only
/// detects surface contact.
///
/// Use ConvexHull when:
/// - You need solid collision detection (e.g., robot safety)
/// - Simplified convex geometry is acceptable
/// - You need to detect objects inside the shape
///
/// Use TriMesh when:
/// - You need exact (possibly concave) mesh geometry
/// - Surface contact detection is sufficient
#[pyclass(module = "py_parry3d._internal", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct ConvexHull {
    hull_vertices: Vec<[f64; 3]>,
    hull_faces: Vec<[u32; 3]>,
}

#[pymethods]
impl ConvexHull {
    /// Create a convex hull from mesh vertices and faces.
    #[staticmethod]
    fn from_mesh(vertices: PyReadonlyArray2<f64>, _faces: PyReadonlyArray2<u32>) -> PyResult<Self> {
        let verts_shape = vertices.shape();

        if verts_shape.len() != 2 || verts_shape[1] != 3 {
            return Err(PyValueError::new_err("vertices must be (N, 3) array"));
        }

        // Convert input vertices to parry's vector type
        let points: Vec<Vec3> = vertices
            .as_slice()?
            .chunks(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect();

        // Compute convex hull using parry3d
        let hull = ConvexPolyhedron::from_convex_hull(&points)
            .ok_or_else(|| PyValueError::new_err("Failed to compute convex hull from points"))?;

        // Extract hull vertices
        let hull_vertices: Vec<[f64; 3]> = hull
            .points()
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();

        // Extract hull faces from topology
        let mut hull_faces: Vec<[u32; 3]> = Vec::new();
        let vertices_adj = hull.vertices_adj_to_face();

        for face in hull.faces() {
            let first = face.first_vertex_or_edge as usize;
            let num = face.num_vertices_or_edges as usize;

            // Get vertices for this face
            let face_vertices: Vec<u32> = (0..num)
                .map(|i| vertices_adj[first + i])
                .collect();

            // Triangulate the face (fan triangulation from first vertex)
            for i in 1..(face_vertices.len() - 1) {
                hull_faces.push([
                    face_vertices[0],
                    face_vertices[i] as u32,
                    face_vertices[i + 1] as u32,
                ]);
            }
        }

        Ok(ConvexHull {
            hull_vertices,
            hull_faces,
        })
    }

    /// Get the convex hull vertices as (N, 3) float64 array.
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.hull_vertices.len();
        let flat: Vec<f64> = self.hull_vertices.iter().flat_map(|v| v.iter().copied()).collect();
        let arr = flat.into_pyarray(py);
        Ok(arr.reshape([n, 3])?)
    }

    /// Get the convex hull faces as (M, 3) uint32 array.
    #[getter]
    fn faces<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let m = self.hull_faces.len();
        let flat: Vec<u32> = self.hull_faces.iter().flat_map(|f| f.iter().copied()).collect();
        let arr = flat.into_pyarray(py);
        Ok(arr.reshape([m, 3])?)
    }

    fn __repr__(&self) -> String {
        format!("ConvexHull(vertices={}, faces={})", self.hull_vertices.len(), self.hull_faces.len())
    }
}

impl ConvexHull {
    fn to_shared_shape(&self) -> (SharedShape, Pose3) {
        let points: Vec<Vec3> = self.hull_vertices
            .iter()
            .map(|v| Vec3::new(v[0], v[1], v[2]))
            .collect();

        // Recreate the convex polyhedron from stored hull vertices
        (
            SharedShape::new(ConvexPolyhedron::from_convex_hull(&points).unwrap()),
            Pose3::IDENTITY,
        )
    }
}

// ============================================================================
// Shape Enum (internal)
// ============================================================================

#[derive(Clone, Serialize, Deserialize)]
enum ShapeData {
    Box(Box),
    Sphere(Sphere),
    Capsule(Capsule),
    Cylinder(Cylinder),
    TriMesh(TriMesh),
    ConvexHull(ConvexHull),
}

impl ShapeData {
    /// The parry shape plus its intrinsic pose offset: the pose that maps the
    /// shape's own frame onto the frame this crate exposes. Only `Cylinder`
    /// needs one (parry's cylinder is Y-aligned, ours is Z-aligned); every
    /// other variant returns the identity.
    fn to_shared_shape(&self) -> (SharedShape, Pose3) {
        match self {
            ShapeData::Box(s) => s.to_shared_shape(),
            ShapeData::Sphere(s) => s.to_shared_shape(),
            ShapeData::Capsule(s) => s.to_shared_shape(),
            ShapeData::Cylinder(s) => s.to_shared_shape(),
            ShapeData::TriMesh(s) => s.to_shared_shape(),
            ShapeData::ConvexHull(s) => s.to_shared_shape(),
        }
    }
}

fn extract_shape(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<ShapeData> {
    if let Ok(s) = obj.extract::<Box>() {
        return Ok(ShapeData::Box(s));
    }
    if let Ok(s) = obj.extract::<Sphere>() {
        return Ok(ShapeData::Sphere(s));
    }
    if let Ok(s) = obj.extract::<Capsule>() {
        return Ok(ShapeData::Capsule(s));
    }
    if let Ok(s) = obj.extract::<Cylinder>() {
        return Ok(ShapeData::Cylinder(s));
    }
    if let Ok(s) = obj.extract::<TriMesh>() {
        return Ok(ShapeData::TriMesh(s));
    }
    if let Ok(s) = obj.extract::<ConvexHull>() {
        return Ok(ShapeData::ConvexHull(s));
    }
    // Check if it's a CollisionObject
    if let Ok(co) = obj.extract::<CollisionObject>() {
        return Ok(co.shape);
    }
    Err(PyTypeError::new_err("Expected a shape type (Box, Sphere, Capsule, Cylinder, TriMesh, ConvexHull) or CollisionObject"))
}

// ============================================================================
// Transform helpers
// ============================================================================

/// Converts a 4x4 homogeneous transform to a parry `Pose3`.
///
/// # Contract
///
/// `m` is indexed `m[row][col]` — **row-major**, matching the NumPy `(4, 4)`
/// arrays callers hand in. It must be a *rigid* transform:
///
/// - the upper-left 3x3 block is a proper rotation: orthonormal columns,
///   determinant +1, no scale, no shear, no reflection;
/// - the last row is `[0, 0, 0, 1]`;
/// - every element is finite (no `NaN`, no infinity).
///
/// Only the 3x3 rotation block and the `m[0..3][3]` translation column are
/// read here; the last row is checked at the API boundary (see below) but not
/// used.
///
/// # This function itself does not validate
///
/// It assumes the contract already holds. Enforcement lives in
/// `validate_rigid_transform`, called from every Python entry point
/// (`extract_transform_4x4` and the `(N, 4, 4)` batch paths), because this
/// function runs once per pose per group inside the Rayon query loop while a
/// caller-supplied transform crosses the boundary exactly once. A non-rigid
/// matrix reaching here would still produce a garbage pose rather than an
/// error: `Rot3::from_mat3` is documented as ill-defined for a 3x3 block
/// carrying scale or shear, and only panics when glam's `glam_assert` feature
/// is on, which it is not in this build.
///
/// # Note on layout
///
/// glam is **column-major** and `Mat3::from_cols` takes columns, so the 3x3
/// block is transposed on the way in: column `j` is built from `m[0][j]`,
/// `m[1][j]`, `m[2][j]`. Getting this backwards inverts every rotation.
fn matrix4_to_isometry(m: &[[f64; 4]; 4]) -> Pose3 {
    let rotation = Mat3::from_cols(
        Vec3::new(m[0][0], m[1][0], m[2][0]),
        Vec3::new(m[0][1], m[1][1], m[2][1]),
        Vec3::new(m[0][2], m[1][2], m[2][2]),
    );
    let translation = Vec3::new(m[0][3], m[1][3], m[2][3]);
    Pose3::from_parts(translation, Rot3::from_mat3(&rotation))
}

/// Tolerance for the orthonormality and determinant checks in
/// [`validate_rigid_transform`].
///
/// Chosen to sit between two magnitudes:
///
/// - **legitimate float drift.** A rotation accumulated through a chain of
///   forward-kinematics products or a quaternion round-trip is orthonormal only
///   to within rounding error; for f64 that is around 1e-15 per operation and
///   stays near 1e-13 even after a long chain. Such matrices are genuine
///   rotations and must be accepted.
/// - **real errors.** A 2x scale is off by 1.0, a shear of 0.5 by 0.5, a
///   reflection has determinant -1. These are off by order 1, not by an epsilon.
///
/// 1e-6 is far above the first and far below the second. Tighten it if callers
/// turn out to feed cleaner matrices than expected; loosening it would start
/// accepting transforms that are genuinely not rotations.
const RIGID_TRANSFORM_TOL: f64 = 1e-6;

/// Tolerance for the bottom row. Effectively an exact check - no legitimate
/// caller passes anything but `[0, 0, 0, 1]` - but expressed as an absolute
/// tolerance so a value that survived a float round-trip is not rejected.
const BOTTOM_ROW_TOL: f64 = 1e-12;

/// Validate that `m` is a rigid transform, as documented on
/// [`matrix4_to_isometry`].
///
/// Checks, in order:
///
/// 1. every one of the 16 entries is finite (no `NaN`, no infinity);
/// 2. the bottom row is `[0, 0, 0, 1]` within [`BOTTOM_ROW_TOL`];
/// 3. the upper-left 3x3 block is a proper rotation - `R^T R = I` and
///    `det(R) = +1` within [`RIGID_TRANSFORM_TOL`]. A determinant of -1 is a
///    reflection, which would mirror the geometry, and is rejected.
///
/// `context` is prefixed to the error message so batch call sites can name the
/// group and index of the offending pose.
///
/// This runs at the Python boundary, not inside `matrix4_to_isometry` - the
/// latter is called once per pose per group inside the Rayon query loop, while
/// every caller-supplied transform passes through a boundary check exactly
/// once. Transforms restored by `CollisionWorld.from_bytes` are not re-checked;
/// they were validated when the world was built.
fn validate_rigid_transform(m: &[[f64; 4]; 4], context: &str) -> PyResult<()> {
    // 1. finiteness
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{context}Transform must be finite, but element [{i}][{j}] is {v}"
                )));
            }
        }
    }

    // 2. bottom row
    let expected = [0.0, 0.0, 0.0, 1.0];
    for j in 0..4 {
        if (m[3][j] - expected[j]).abs() > BOTTOM_ROW_TOL {
            return Err(PyValueError::new_err(format!(
                "{}Transform must have bottom row [0, 0, 0, 1], but got [{}, {}, {}, {}]",
                context, m[3][0], m[3][1], m[3][2], m[3][3]
            )));
        }
    }

    // 3. proper rotation: R^T R = I and det(R) = +1
    let r = [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ];
    for a in 0..3 {
        for b in 0..3 {
            // (R^T R)[a][b] = dot(column a, column b)
            let dot = r[0][a] * r[0][b] + r[1][a] * r[1][b] + r[2][a] * r[2][b];
            let target = if a == b { 1.0 } else { 0.0 };
            if (dot - target).abs() > RIGID_TRANSFORM_TOL {
                return Err(PyValueError::new_err(format!(
                    "{context}Transform rotation block must be orthonormal (no scale or shear): \
                     (R^T R)[{a}][{b}] = {dot}, expected {target} (tolerance {RIGID_TRANSFORM_TOL:e})"
                )));
            }
        }
    }
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    if (det - 1.0).abs() > RIGID_TRANSFORM_TOL {
        return Err(PyValueError::new_err(format!(
            "{context}Transform rotation block must have determinant +1, but got {det} \
             (tolerance {RIGID_TRANSFORM_TOL:e}); a determinant of -1 is a reflection"
        )));
    }

    Ok(())
}

/// Convert the row-major `(N, 4, 4)` data of group `name` into matrices,
/// validating every one, in parallel.
///
/// On failure the error names the group and the lowest invalid index, exactly
/// as a serial scan would. The context string costs about 200 ns per matrix to
/// format, more than the check itself, so it is only built for the matrix that
/// fails.
fn batch_transforms(slice: &[f64], name: &str) -> PyResult<Vec<[[f64; 4]; 4]>> {
    let tfs: Vec<[[f64; 4]; 4]> = slice
        .par_chunks_exact(16)
        .map(|c| [
            [c[0], c[1], c[2], c[3]],
            [c[4], c[5], c[6], c[7]],
            [c[8], c[9], c[10], c[11]],
            [c[12], c[13], c[14], c[15]],
        ])
        .collect();
    if let Some(i) = tfs
        .par_iter()
        .position_first(|m| validate_rigid_transform(m, "").is_err())
    {
        validate_rigid_transform(&tfs[i], &format!("Transform for '{}' at index {}: ", name, i))?;
    }
    Ok(tfs)
}

fn extract_transform_4x4(arr: &PyReadonlyArray2<f64>) -> PyResult<[[f64; 4]; 4]> {
    let shape = arr.shape();
    if shape != [4, 4] {
        return Err(PyValueError::new_err("Transform must be (4, 4) array"));
    }
    let slice = arr.as_slice()?;
    let m = [
        [slice[0], slice[1], slice[2], slice[3]],
        [slice[4], slice[5], slice[6], slice[7]],
        [slice[8], slice[9], slice[10], slice[11]],
        [slice[12], slice[13], slice[14], slice[15]],
    ];
    validate_rigid_transform(&m, "")?;
    Ok(m)
}

// ============================================================================
// CollisionObject
// ============================================================================

/// A shape with an optional local transform.
#[pyclass(module = "py_parry3d._internal", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct CollisionObject {
    shape: ShapeData,
    transform: [[f64; 4]; 4],
}

#[pymethods]
impl CollisionObject {
    #[new]
    #[pyo3(signature = (shape, transform=None))]
    fn new(py: Python<'_>, shape: &Bound<'_, PyAny>, transform: Option<PyReadonlyArray2<f64>>) -> PyResult<Self> {
        let shape_data = extract_shape(py, shape)?;

        let tf = if let Some(t) = transform {
            extract_transform_4x4(&t)?
        } else {
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        };

        Ok(CollisionObject {
            shape: shape_data,
            transform: tf,
        })
    }

    fn __repr__(&self) -> String {
        format!("CollisionObject(...)")
    }
}

impl CollisionObject {
    fn to_isometry(&self) -> Pose3 {
        matrix4_to_isometry(&self.transform)
    }
}

// ============================================================================
// CollisionGroup
// ============================================================================

/// A named group of collision objects sharing the same transform.
#[pyclass(module = "py_parry3d._internal", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct CollisionGroup {
    #[pyo3(get)]
    name: String,
    objects: Vec<CollisionObject>,
    #[pyo3(get)]
    is_static: bool,
    static_transform: Option<[[f64; 4]; 4]>,
    #[serde(skip)]
    cached_shape: Option<Arc<(SharedShape, Pose3)>>,
}

#[pymethods]
impl CollisionGroup {
    #[new]
    #[pyo3(signature = (name, objects, is_static=false, transform=None))]
    fn new(
        py: Python<'_>,
        name: String,
        objects: &Bound<'_, PyList>,
        is_static: Option<bool>,
        transform: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        let is_static = is_static.unwrap_or(false);

        let static_tf = if let Some(t) = transform {
            Some(extract_transform_4x4(&t)?)
        } else {
            None
        };

        if is_static && static_tf.is_none() {
            return Err(PyValueError::new_err("Static groups require a transform"));
        }

        let mut collision_objects = Vec::new();
        for item in objects.iter() {
            // Check if it's a CollisionObject
            if let Ok(co) = item.extract::<CollisionObject>() {
                collision_objects.push(co);
            } else {
                // Try to interpret as a bare shape
                let shape_data = extract_shape(py, &item)?;
                collision_objects.push(CollisionObject {
                    shape: shape_data,
                    transform: [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                });
            }
        }

        Ok(CollisionGroup {
            name,
            objects: collision_objects,
            is_static,
            static_transform: static_tf,
            cached_shape: None,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CollisionGroup(name='{}', objects={}, static={})",
            self.name,
            self.objects.len(),
            self.is_static
        )
    }

    fn __len__(&self) -> usize {
        self.objects.len()
    }
}

impl CollisionGroup {
    /// The group's collision shape plus the pose to compose onto the group
    /// pose before querying it.
    ///
    /// A multi-object group is a parry `Compound`, which carries every object's
    /// local pose itself, so the companion pose is the identity. A lone
    /// object is kept as its bare shape — parry's query dispatch does not
    /// support a `TriMesh` or `ConvexPolyhedron` inside a `Compound` — and its
    /// local pose is returned alongside so the query can apply it. Dropping
    /// it, as the shortcut used to, tested the shape centred on the group frame
    /// (#1).
    ///
    /// A shape may also carry an intrinsic pose offset of its own (a
    /// `Cylinder` does, to reorient parry's Y-aligned cylinder to Z); it is
    /// composed onto the object's local pose in both branches.
    fn build_shape(&mut self) -> Arc<(SharedShape, Pose3)> {
        if let Some(ref cached) = self.cached_shape {
            return cached.clone();
        }

        let result = if self.objects.len() == 1 {
            let (shape, shape_offset) = self.objects[0].shape.to_shared_shape();
            Arc::new((shape, self.objects[0].to_isometry() * shape_offset))
        } else {
            let shapes: Vec<(Pose3, SharedShape)> = self.objects
                .iter()
                .map(|o| {
                    let (shape, shape_offset) = o.shape.to_shared_shape();
                    (o.to_isometry() * shape_offset, shape)
                })
                .collect();
            Arc::new((SharedShape::new(Compound::new(shapes)), Pose3::IDENTITY))
        };
        self.cached_shape = Some(result.clone());
        result
    }

    fn get_static_isometry(&self) -> Option<Pose3> {
        self.static_transform.as_ref().map(|t| matrix4_to_isometry(t))
    }
}

// ============================================================================
// CollisionWorld
// ============================================================================

#[derive(Serialize, Deserialize)]
struct CollisionWorldData {
    groups: Vec<CollisionGroup>,
    group_indices: HashMap<String, usize>,
    dynamic_group_names: Vec<String>,
    static_group_names: Vec<String>,
}

/// Container for all collision groups.
#[pyclass(module = "py_parry3d._internal")]
pub struct CollisionWorld {
    data: CollisionWorldData,
    // Cached shapes (not serialized, rebuilt on load)
    shapes: Vec<Arc<(SharedShape, Pose3)>>,
}

#[pymethods]
impl CollisionWorld {
    #[new]
    fn new(_py: Python<'_>, groups: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut group_vec: Vec<CollisionGroup> = Vec::new();
        let mut group_indices: HashMap<String, usize> = HashMap::new();
        let mut dynamic_names: Vec<String> = Vec::new();
        let mut static_names: Vec<String> = Vec::new();

        for (idx, item) in groups.iter().enumerate() {
            let group: CollisionGroup = item.extract()?;

            if group_indices.contains_key(&group.name) {
                return Err(PyValueError::new_err(format!(
                    "Duplicate group name: '{}'",
                    group.name
                )));
            }

            group_indices.insert(group.name.clone(), idx);

            if group.is_static {
                static_names.push(group.name.clone());
            } else {
                dynamic_names.push(group.name.clone());
            }

            group_vec.push(group);
        }

        // Build cached shapes
        let mut shapes = Vec::new();
        for group in &mut group_vec {
            shapes.push(group.build_shape());
        }

        Ok(CollisionWorld {
            data: CollisionWorldData {
                groups: group_vec,
                group_indices,
                dynamic_group_names: dynamic_names,
                static_group_names: static_names,
            },
            shapes,
        })
    }

    #[getter]
    fn groups(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for group in &self.data.groups {
            dict.set_item(&group.name, group.clone().into_pyobject(py)?)?;
        }
        Ok(dict.into())
    }

    #[getter]
    fn dynamic_groups(&self) -> Vec<String> {
        self.data.dynamic_group_names.clone()
    }

    #[getter]
    fn static_groups(&self) -> Vec<String> {
        self.data.static_group_names.clone()
    }

    fn __len__(&self) -> usize {
        self.data.groups.len()
    }

    /// Check collisions for given transforms and pairs.
    ///
    /// transforms: dict mapping group name to (N, 4, 4) or (4, 4) array
    /// pairs: list of (group_name, group_name, min_distance) tuples
    ///
    /// Returns: (N, n_pairs) or (n_pairs,) bool array
    fn check<'py>(
        &self,
        py: Python<'py>,
        transforms: &Bound<'py, PyDict>,
        pairs: &Bound<'py, PyList>,
    ) -> PyResult<Py<PyAny>> {
        let query = self.prepare_query(transforms, pairs)?;
        let n = query.n;
        let n_pairs = query.pairs.len();

        // Perform collision checking in parallel without the GIL, filling
        // row `pose_idx` of one flat (n, n_pairs) buffer.
        let mut flat = vec![false; n * n_pairs];
        // par_chunks_mut panics on a chunk size of 0; with no pairs there is nothing to fill.
        if n_pairs > 0 {
            py.detach(|| {
                flat.par_chunks_mut(n_pairs).enumerate().for_each_init(
                    || query.pose_scratch(),
                    |poses, (pose_idx, row)| {
                        query.fill_poses(pose_idx, poses);
                        query.write_pair_results(poses, row);
                    },
                )
            });
        }

        // Convert to numpy array
        if n == 1 {
            // Return (n_pairs,) array
            let result = flat.into_pyarray(py);
            Ok(result.into_any().unbind())
        } else {
            // Return (N, n_pairs) array
            let arr = flat.into_pyarray(py);
            let reshaped = arr.reshape([n, n_pairs])?;
            Ok(reshaped.into_any().unbind())
        }
    }

    /// Check for any collision, returning early on the first hit any thread finds.
    ///
    /// Returns: Optional[int] - index of a pose with collision (not necessarily
    /// the lowest), or None
    fn check_any<'py>(
        &self,
        py: Python<'py>,
        transforms: &Bound<'py, PyDict>,
        pairs: &Bound<'py, PyList>,
    ) -> PyResult<Option<usize>> {
        let query = self.prepare_query(transforms, pairs)?;
        Ok(py.detach(|| {
            query
                .pose_hits()
                .find_any(|&(_, hit)| hit)
                .map(|(pose_idx, _)| pose_idx)
        }))
    }

    /// Find the lowest-index colliding pose and its per-pair results.
    ///
    /// Takes the same arguments as `check`. Deterministic regardless of the
    /// number of threads.
    ///
    /// Returns: Optional[tuple[int, (n_pairs,) bool array]] - the first pose
    /// with any collision and which pairs collide there, or None
    fn check_first<'py>(
        &self,
        py: Python<'py>,
        transforms: &Bound<'py, PyDict>,
        pairs: &Bound<'py, PyList>,
    ) -> PyResult<Option<(usize, Bound<'py, PyArray1<bool>>)>> {
        let query = self.prepare_query(transforms, pairs)?;
        // find_first returns the lowest matching index, unlike find_any.
        let first = py.detach(|| {
            query
                .pose_hits()
                .find_first(|&(_, hit)| hit)
                .map(|(pose_idx, _)| pose_idx)
        });

        Ok(first.map(|pose_idx| {
            let mut poses = query.pose_scratch();
            query.fill_poses(pose_idx, &mut poses);
            (pose_idx, query.pair_results(&poses).into_pyarray(py))
        }))
    }

    /// Serialize to bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = bincode::serialize(&self.data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Deserialize from bytes.
    #[staticmethod]
    fn from_bytes(_py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = data.as_bytes();
        let mut world_data: CollisionWorldData = bincode::deserialize(bytes)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        // Rebuild cached shapes
        let mut shapes = Vec::new();
        for group in &mut world_data.groups {
            shapes.push(group.build_shape());
        }

        Ok(CollisionWorld {
            data: world_data,
            shapes,
        })
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.to_bytes(py)
    }

    fn __setstate__(&mut self, _py: Python<'_>, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let bytes = state.as_bytes();
        let mut world_data: CollisionWorldData = bincode::deserialize(bytes)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        // Rebuild cached shapes
        let mut shapes = Vec::new();
        for group in &mut world_data.groups {
            shapes.push(group.build_shape());
        }

        self.data = world_data;
        self.shapes = shapes;
        Ok(())
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Py<PyAny>, (Bound<'py, PyBytes>,))> {
        let cls = py.get_type::<CollisionWorld>();
        let state = self.to_bytes(py)?;
        Ok((cls.getattr("from_bytes")?.into(), (state,)))
    }
}

// Not #[pymethods]: these helpers must not be callable from Python.
impl CollisionWorld {
    /// Parse and validate the arguments shared by `check`, `check_any` and
    /// `check_first`, resolving every group name to an index.
    fn prepare_query(
        &self,
        transforms: &Bound<'_, PyDict>,
        pairs: &Bound<'_, PyList>,
    ) -> PyResult<PreparedQuery<'_>> {
        let pairs = self.resolve_pairs(&parse_pairs(pairs)?)?;
        let (n, dynamic) = self.parse_transforms(transforms)?;

        // Static poses are fixed, so they are composed once rather than per pose.
        let base_poses = self
            .data
            .groups
            .iter()
            .zip(self.shapes.iter())
            .map(|(g, s)| {
                if g.is_static {
                    g.get_static_isometry().map(|iso| iso * s.1)
                } else {
                    None
                }
            })
            .collect();

        Ok(PreparedQuery {
            n,
            pairs,
            dynamic,
            shapes: &self.shapes,
            base_poses,
        })
    }

    /// Map `(group_a, group_b, min_distance)` names to group indices.
    fn resolve_pairs(&self, pairs: &[(String, String, f64)]) -> PyResult<Vec<(usize, usize, f64)>> {
        let index_of = |name: &str| {
            self.data
                .group_indices
                .get(name)
                .copied()
                .ok_or_else(|| PyValueError::new_err(format!("Unknown group: '{}'", name)))
        };
        pairs
            .iter()
            .map(|(a, b, min_dist)| Ok((index_of(a)?, index_of(b)?, *min_dist)))
            .collect()
    }

    /// Read and validate the transform of every dynamic group.
    ///
    /// Returns the batch size (1 for `(4, 4)` transforms) and, per dynamic
    /// group, its index and one matrix per pose.
    fn parse_transforms(
        &self,
        transforms: &Bound<'_, PyDict>,
    ) -> PyResult<(usize, DynamicTransforms)> {
        let mut dynamic = Vec::with_capacity(self.data.dynamic_group_names.len());
        let mut batch_size: Option<usize> = None;

        for dynamic_name in &self.data.dynamic_group_names {
            let Some(arr_any) = transforms.get_item(dynamic_name)? else {
                return Err(PyValueError::new_err(format!(
                    "Missing transform for dynamic group: '{}'",
                    dynamic_name
                )));
            };

            let tfs = if let Ok(arr4) = arr_any.extract::<PyReadonlyArray3<f64>>() {
                // (N, 4, 4) batch
                let shape = arr4.shape();
                if shape[1] != 4 || shape[2] != 4 {
                    return Err(PyValueError::new_err(format!(
                        "Transform for '{}' must be (N, 4, 4) or (4, 4)",
                        dynamic_name
                    )));
                }
                let n = shape[0];

                if let Some(bs) = batch_size {
                    if bs != n {
                        return Err(PyValueError::new_err(
                            "All transform arrays must have same batch size"
                        ));
                    }
                } else {
                    batch_size = Some(n);
                }

                let slice = arr4.as_slice()?;
                transforms.py().detach(|| batch_transforms(slice, dynamic_name))?
            } else if let Ok(arr2) = arr_any.extract::<PyReadonlyArray2<f64>>() {
                // (4, 4) single
                let tf = extract_transform_4x4(&arr2)?;
                if batch_size.is_none() {
                    batch_size = Some(1);
                } else if batch_size != Some(1) {
                    return Err(PyValueError::new_err(
                        "Mixing single and batch transforms"
                    ));
                }
                vec![tf]
            } else {
                return Err(PyValueError::new_err(format!(
                    "Transform for '{}' must be numpy array",
                    dynamic_name
                )));
            };
            dynamic.push((self.data.group_indices[dynamic_name], tfs));
        }

        Ok((batch_size.unwrap_or(1), dynamic))
    }
}

/// Per dynamic group: its index and one matrix per pose.
type DynamicTransforms = Vec<(usize, Vec<[[f64; 4]; 4]>)>;

/// A validated collision query with all group names resolved to indices.
struct PreparedQuery<'a> {
    /// Number of poses.
    n: usize,
    /// `(group_a, group_b, min_distance)` as group indices, in `pairs` order.
    pairs: Vec<(usize, usize, f64)>,
    dynamic: DynamicTransforms,
    /// Shape and local pose offset per group.
    shapes: &'a [Arc<(SharedShape, Pose3)>],
    /// Pose per group with static groups already filled in.
    base_poses: Vec<Option<Pose3>>,
}

impl PreparedQuery<'_> {
    /// Per-thread buffer for `fill_poses`, so the pose loop does not allocate.
    fn pose_scratch(&self) -> Vec<Option<Pose3>> {
        self.base_poses.clone()
    }

    /// Write every dynamic group's pose at `pose_idx` into `poses`.
    fn fill_poses(&self, pose_idx: usize, poses: &mut [Option<Pose3>]) {
        for (group_idx, tfs) in &self.dynamic {
            // A lone object's local transform lives in the offset, not in the shape.
            let offset = self.shapes[*group_idx].1;
            poses[*group_idx] = Some(matrix4_to_isometry(&tfs[pose_idx]) * offset);
        }
    }

    fn pair_collides(&self, poses: &[Option<Pose3>], (idx_a, idx_b, min_dist): (usize, usize, f64)) -> bool {
        let (Some(iso_a), Some(iso_b)) = (&poses[idx_a], &poses[idx_b]) else {
            return false;
        };
        let shape_a = &self.shapes[idx_a].0;
        let shape_b = &self.shapes[idx_b].0;

        if min_dist > 0.0 {
            // Use distance query with threshold
            let dist = query::distance(iso_a, shape_a.as_ref(), iso_b, shape_b.as_ref())
                .unwrap_or(f64::MAX);
            dist < min_dist
        } else {
            // Use faster intersection test
            query::intersection_test(iso_a, shape_a.as_ref(), iso_b, shape_b.as_ref())
                .unwrap_or(false)
        }
    }

    /// Whether any pair collides, stopping at the first one that does.
    fn any_collides(&self, poses: &[Option<Pose3>]) -> bool {
        self.pairs.iter().any(|&pair| self.pair_collides(poses, pair))
    }

    /// One bool per pair, in `pairs` order.
    fn pair_results(&self, poses: &[Option<Pose3>]) -> Vec<bool> {
        self.pairs.iter().map(|&pair| self.pair_collides(poses, pair)).collect()
    }

    /// `pair_results` written into `row`, one bool per pair, without allocating.
    fn write_pair_results(&self, poses: &[Option<Pose3>], row: &mut [bool]) {
        for (out, &pair) in row.iter_mut().zip(&self.pairs) {
            *out = self.pair_collides(poses, pair);
        }
    }

    /// `(pose_idx, any pair collides)` for every pose, reusing one pose buffer per job.
    fn pose_hits(&self) -> impl ParallelIterator<Item = (usize, bool)> + '_ {
        (0..self.n).into_par_iter().map_init(
            || self.pose_scratch(),
            |poses, pose_idx| {
                self.fill_poses(pose_idx, poses);
                (pose_idx, self.any_collides(poses))
            },
        )
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse a list of `(group_a, group_b, min_distance)` 3-tuples.
///
/// Validates that every element is a tuple of exactly three items with the
/// right element types, so a malformed `pairs` argument names the offending
/// index instead of raising an opaque `IndexError` (too few items) or being
/// silently truncated (too many items).
fn parse_pairs(pairs: &Bound<'_, PyList>) -> PyResult<Vec<(String, String, f64)>> {
    pairs
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let tuple = item.cast::<pyo3::types::PyTuple>().map_err(|_| {
                PyValueError::new_err(format!(
                    "pairs[{}]: expected a (group_a, group_b, min_distance) 3-tuple, got '{}'",
                    i,
                    type_name_of(&item)
                ))
            })?;
            if tuple.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "pairs[{}]: expected a (group_a, group_b, min_distance) 3-tuple, got {} elements",
                    i,
                    tuple.len()
                )));
            }
            let a_obj = tuple.get_item(0)?;
            let a: String = a_obj.extract().map_err(|_| {
                PyTypeError::new_err(format!(
                    "pairs[{}]: group_a must be a str, got '{}'",
                    i,
                    type_name_of(&a_obj)
                ))
            })?;
            let b_obj = tuple.get_item(1)?;
            let b: String = b_obj.extract().map_err(|_| {
                PyTypeError::new_err(format!(
                    "pairs[{}]: group_b must be a str, got '{}'",
                    i,
                    type_name_of(&b_obj)
                ))
            })?;
            let d_obj = tuple.get_item(2)?;
            let min_dist: f64 = d_obj.extract().map_err(|_| {
                PyTypeError::new_err(format!(
                    "pairs[{}]: min_distance must be a float, got '{}'",
                    i,
                    type_name_of(&d_obj)
                ))
            })?;
            Ok((a, b, min_dist))
        })
        .collect()
}

/// Best-effort type name for use in error messages.
fn type_name_of(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map_or_else(|_| "unknown".to_string(), |n| n.to_string())
}

/// Generate all pairs between groups, optionally skipping adjacent indices.
#[pyfunction]
#[pyo3(signature = (groups, skip_adjacent=0, min_distance=0.0))]
fn all_pairs(groups: &Bound<'_, PyList>, skip_adjacent: usize, min_distance: f64) -> PyResult<Vec<(String, String, f64)>> {
    let names: Vec<String> = groups.iter().map(|item| item.extract()).collect::<PyResult<_>>()?;
    let n = names.len();
    let mut pairs = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if j - i > skip_adjacent {
                pairs.push((names[i].clone(), names[j].clone(), min_distance));
            }
        }
    }

    Ok(pairs)
}

/// Generate pairs between one set of groups and another group/groups.
#[pyfunction]
#[pyo3(signature = (groups, other, min_distance=0.0))]
fn pairs_vs(_py: Python<'_>, groups: &Bound<'_, PyList>, other: &Bound<'_, PyAny>, min_distance: f64) -> PyResult<Vec<(String, String, f64)>> {
    let names: Vec<String> = groups.iter().map(|item| item.extract()).collect::<PyResult<_>>()?;

    let others: Vec<String> = if let Ok(s) = other.extract::<String>() {
        vec![s]
    } else if let Ok(list) = other.cast::<PyList>() {
        list.iter().map(|item| item.extract()).collect::<PyResult<_>>()?
    } else {
        return Err(PyTypeError::new_err("other must be string or list of strings"));
    };

    let mut pairs = Vec::new();
    for name in &names {
        for other_name in &others {
            pairs.push((name.clone(), other_name.clone(), min_distance));
        }
    }

    Ok(pairs)
}

/// Create a 4x4 transform matrix from rotation and/or translation.
#[pyfunction]
#[pyo3(signature = (rotation=None, translation=None))]
fn transform<'py>(
    py: Python<'py>,
    rotation: Option<&Bound<'py, PyAny>>,
    translation: Option<[f64; 3]>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut mat = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    // Handle rotation if provided
    if let Some(rot) = rotation {
        // Try to call as_matrix() on the rotation object (scipy.spatial.transform.Rotation)
        if let Ok(as_matrix_fn) = rot.getattr("as_matrix") {
            let rot_mat = as_matrix_fn.call0()?;
            let rot_arr: PyReadonlyArray2<f64> = rot_mat.extract()?;
            let slice = rot_arr.as_slice()?;

            mat[0][0] = slice[0]; mat[0][1] = slice[1]; mat[0][2] = slice[2];
            mat[1][0] = slice[3]; mat[1][1] = slice[4]; mat[1][2] = slice[5];
            mat[2][0] = slice[6]; mat[2][1] = slice[7]; mat[2][2] = slice[8];
        } else {
            return Err(PyTypeError::new_err("rotation must have as_matrix() method"));
        }
    }

    // Handle translation if provided
    if let Some(t) = translation {
        mat[0][3] = t[0];
        mat[1][3] = t[1];
        mat[2][3] = t[2];
    }

    let flat: Vec<f64> = mat.iter().flat_map(|row| row.iter().copied()).collect();
    let arr = flat.into_pyarray(py);
    Ok(arr.reshape([4, 4])?)
}

/// Set the number of threads for parallel operations.
///
/// Must be called BEFORE the first parallel operation (check/check_any/check_first).
/// Returns True if successful, False if thread pool was already initialized.
#[pyfunction]
fn set_num_threads(n: usize) -> bool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .is_ok()
}

/// Get the number of threads for parallel operations.
#[pyfunction]
fn get_num_threads() -> usize {
    rayon::current_num_threads()
}

// ============================================================================
// Module
// ============================================================================

#[pymodule]
#[pyo3(name = "_internal")]
fn py_parry3d(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Shapes
    m.add_class::<Box>()?;
    m.add_class::<Sphere>()?;
    m.add_class::<Capsule>()?;
    m.add_class::<Cylinder>()?;
    m.add_class::<TriMesh>()?;
    m.add_class::<ConvexHull>()?;

    // Core types
    m.add_class::<CollisionObject>()?;
    m.add_class::<CollisionGroup>()?;
    m.add_class::<CollisionWorld>()?;

    // Helper functions
    m.add_function(wrap_pyfunction!(all_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(pairs_vs, m)?)?;
    m.add_function(wrap_pyfunction!(transform, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;

    Ok(())
}
