use nalgebra::{Isometry3, Matrix4, Point3, Translation3, UnitQuaternion, Vector3};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods, IntoPyArray};
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
#[pyclass(module = "py_parry3d._internal")]
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
    fn to_shared_shape(&self) -> SharedShape {
        SharedShape::new(Cuboid::new(Vector3::new(
            self.half_extents[0],
            self.half_extents[1],
            self.half_extents[2],
        )))
    }
}

/// A sphere shape with given radius.
#[pyclass(module = "py_parry3d._internal")]
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
    fn to_shared_shape(&self) -> SharedShape {
        SharedShape::new(Ball::new(self.radius))
    }
}

/// A capsule shape (cylinder with hemispherical caps) along Z axis.
#[pyclass(module = "py_parry3d._internal")]
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
    fn to_shared_shape(&self) -> SharedShape {
        // Z-axis aligned capsule
        let capsule = ParryCapsule::new_z(self.half_height, self.radius);
        SharedShape::new(capsule)
    }
}

/// A cylinder shape along Z axis.
#[pyclass(module = "py_parry3d._internal")]
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
    fn to_shared_shape(&self) -> SharedShape {
        // parry3d Cylinder is along Y axis, rotate to Z axis
        // Rotate 90 degrees around X axis: Y -> Z
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_2);
        let iso = Isometry3::from_parts(Translation3::identity(), rotation);
        let cylinder = ParryCylinder::new(self.half_height, self.radius);
        SharedShape::new(Compound::new(vec![(iso, SharedShape::new(cylinder))]))
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
#[pyclass(module = "py_parry3d._internal")]
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
    fn to_shared_shape(&self) -> SharedShape {
        let points: Vec<Point3<f64>> = self.vertices
            .iter()
            .map(|v| Point3::new(v[0], v[1], v[2]))
            .collect();

        let trimesh = ParryTriMesh::new(points, self.faces.clone())
            .expect("Failed to create TriMesh");
        SharedShape::new(trimesh)
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
#[pyclass(module = "py_parry3d._internal")]
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

        // Convert input vertices to Point3
        let points: Vec<Point3<f64>> = vertices
            .as_slice()?
            .chunks(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
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
    fn to_shared_shape(&self) -> SharedShape {
        let points: Vec<Point3<f64>> = self.hull_vertices
            .iter()
            .map(|v| Point3::new(v[0], v[1], v[2]))
            .collect();

        // Recreate the convex polyhedron from stored hull vertices
        SharedShape::new(ConvexPolyhedron::from_convex_hull(&points).unwrap())
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
    fn to_shared_shape(&self) -> SharedShape {
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

fn matrix4_to_isometry(m: &[[f64; 4]; 4]) -> Isometry3<f64> {
    let rotation = nalgebra::Rotation3::from_matrix_unchecked(Matrix4::new(
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ).fixed_view::<3, 3>(0, 0).into());

    let translation = Translation3::new(m[0][3], m[1][3], m[2][3]);
    Isometry3::from_parts(translation, UnitQuaternion::from_rotation_matrix(&rotation))
}

fn extract_transform_4x4(arr: &PyReadonlyArray2<f64>) -> PyResult<[[f64; 4]; 4]> {
    let shape = arr.shape();
    if shape != [4, 4] {
        return Err(PyValueError::new_err("Transform must be (4, 4) array"));
    }
    let slice = arr.as_slice()?;
    Ok([
        [slice[0], slice[1], slice[2], slice[3]],
        [slice[4], slice[5], slice[6], slice[7]],
        [slice[8], slice[9], slice[10], slice[11]],
        [slice[12], slice[13], slice[14], slice[15]],
    ])
}

// ============================================================================
// CollisionObject
// ============================================================================

/// A shape with an optional local transform.
#[pyclass(module = "py_parry3d._internal")]
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
    fn to_isometry(&self) -> Isometry3<f64> {
        matrix4_to_isometry(&self.transform)
    }
}

// ============================================================================
// CollisionGroup
// ============================================================================

/// A named group of collision objects sharing the same transform.
#[pyclass(module = "py_parry3d._internal")]
#[derive(Clone, Serialize, Deserialize)]
pub struct CollisionGroup {
    #[pyo3(get)]
    name: String,
    objects: Vec<CollisionObject>,
    #[pyo3(get)]
    is_static: bool,
    static_transform: Option<[[f64; 4]; 4]>,
    #[serde(skip)]
    cached_shape: Option<Arc<(SharedShape, Vec<Isometry3<f64>>)>>,
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
    fn build_shape(&mut self) -> Arc<(SharedShape, Vec<Isometry3<f64>>)> {
        if let Some(ref cached) = self.cached_shape {
            return cached.clone();
        }

        let local_isometries: Vec<Isometry3<f64>> = self.objects
            .iter()
            .map(|o| o.to_isometry())
            .collect();

        let shape = if self.objects.len() == 1 {
            self.objects[0].shape.to_shared_shape()
        } else {
            // Create compound shape
            let shapes: Vec<(Isometry3<f64>, SharedShape)> = self.objects
                .iter()
                .map(|o| (o.to_isometry(), o.shape.to_shared_shape()))
                .collect();
            SharedShape::new(Compound::new(shapes))
        };

        let result = Arc::new((shape, local_isometries));
        self.cached_shape = Some(result.clone());
        result
    }

    fn get_static_isometry(&self) -> Option<Isometry3<f64>> {
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
    shapes: Vec<Arc<(SharedShape, Vec<Isometry3<f64>>)>>,
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
        // Parse pairs (3-tuples with min_distance)
        let pair_vec: Vec<(String, String, f64)> = pairs
            .iter()
            .map(|item| {
                let tuple = item.cast::<pyo3::types::PyTuple>()?;
                let a: String = tuple.get_item(0)?.extract()?;
                let b: String = tuple.get_item(1)?.extract()?;
                let min_dist: f64 = tuple.get_item(2)?.extract()?;
                Ok((a, b, min_dist))
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Validate pair group names
        for (a, b, _) in &pair_vec {
            if !self.data.group_indices.contains_key(a) {
                return Err(PyValueError::new_err(format!("Unknown group: '{}'", a)));
            }
            if !self.data.group_indices.contains_key(b) {
                return Err(PyValueError::new_err(format!("Unknown group: '{}'", b)));
            }
        }

        // Parse transforms and determine batch size
        let mut transform_arrays: HashMap<String, Vec<[[f64; 4]; 4]>> = HashMap::new();
        let mut batch_size: Option<usize> = None;

        for dynamic_name in &self.data.dynamic_group_names {
            let arr_obj = transforms.get_item(dynamic_name)?;
            if arr_obj.is_none() {
                return Err(PyValueError::new_err(format!(
                    "Missing transform for dynamic group: '{}'",
                    dynamic_name
                )));
            }
            let arr_obj = arr_obj.unwrap();

            // Try to interpret as numpy array
            let arr_any = arr_obj;

            // Check dimensions
            if let Ok(arr4) = arr_any.extract::<PyReadonlyArray3<f64>>() {
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
                let mut tfs = Vec::with_capacity(n);
                for i in 0..n {
                    let base = i * 16;
                    tfs.push([
                        [slice[base], slice[base+1], slice[base+2], slice[base+3]],
                        [slice[base+4], slice[base+5], slice[base+6], slice[base+7]],
                        [slice[base+8], slice[base+9], slice[base+10], slice[base+11]],
                        [slice[base+12], slice[base+13], slice[base+14], slice[base+15]],
                    ]);
                }
                transform_arrays.insert(dynamic_name.clone(), tfs);
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
                transform_arrays.insert(dynamic_name.clone(), vec![tf]);
            } else {
                return Err(PyValueError::new_err(format!(
                    "Transform for '{}' must be numpy array",
                    dynamic_name
                )));
            }
        }

        let n = batch_size.unwrap_or(1);
        let n_pairs = pair_vec.len();

        // Convert pairs to indices for faster access
        let pair_indices: Vec<(usize, usize, f64)> = pair_vec
            .iter()
            .map(|(a, b, min_dist)| {
                (self.data.group_indices[a], self.data.group_indices[b], *min_dist)
            })
            .collect();

        // Prepare group data for collision checking
        struct GroupCheckData {
            shape: SharedShape,
            is_static: bool,
            static_isometry: Option<Isometry3<f64>>,
        }

        let group_data: Vec<GroupCheckData> = self.data.groups
            .iter()
            .zip(self.shapes.iter())
            .map(|(g, s)| GroupCheckData {
                shape: s.0.clone(),
                is_static: g.is_static,
                static_isometry: g.get_static_isometry(),
            })
            .collect();

        // Perform collision checking in parallel
        let results: Vec<Vec<bool>> = (0..n)
            .into_par_iter()
            .map(|pose_idx| {
                // Build isometries for this pose
                let mut isometries: Vec<Option<Isometry3<f64>>> = vec![None; self.data.groups.len()];

                for (name, tfs) in &transform_arrays {
                    let group_idx = self.data.group_indices[name];
                    isometries[group_idx] = Some(matrix4_to_isometry(&tfs[pose_idx]));
                }

                // Set static isometries
                for (idx, gd) in group_data.iter().enumerate() {
                    if gd.is_static {
                        isometries[idx] = gd.static_isometry.clone();
                    }
                }

                // Check each pair
                pair_indices
                    .iter()
                    .map(|&(idx_a, idx_b, min_dist)| {
                        let iso_a = match &isometries[idx_a] {
                            Some(iso) => iso,
                            None => return false, // NaN or missing
                        };
                        let iso_b = match &isometries[idx_b] {
                            Some(iso) => iso,
                            None => return false,
                        };

                        let shape_a = &group_data[idx_a].shape;
                        let shape_b = &group_data[idx_b].shape;

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
                    })
                    .collect()
            })
            .collect();

        // Convert to numpy array
        if n == 1 {
            // Return (n_pairs,) array
            let result = results[0].clone().into_pyarray(py);
            Ok(result.into_any().unbind())
        } else {
            // Return (N, n_pairs) array - flatten and reshape
            let flat: Vec<bool> = results.into_iter().flatten().collect();
            let arr = flat.into_pyarray(py);
            let reshaped = arr.reshape([n, n_pairs])?;
            Ok(reshaped.into_any().unbind())
        }
    }

    /// Check for any collision, returning early on first hit.
    ///
    /// Returns: Optional[int] - index of first pose with collision, or None
    fn check_any<'py>(
        &self,
        _py: Python<'py>,
        transforms: &Bound<'py, PyDict>,
        pairs: &Bound<'py, PyList>,
    ) -> PyResult<Option<usize>> {
        // Parse pairs (3-tuples with min_distance)
        let pair_vec: Vec<(String, String, f64)> = pairs
            .iter()
            .map(|item| {
                let tuple = item.cast::<pyo3::types::PyTuple>()?;
                let a: String = tuple.get_item(0)?.extract()?;
                let b: String = tuple.get_item(1)?.extract()?;
                let min_dist: f64 = tuple.get_item(2)?.extract()?;
                Ok((a, b, min_dist))
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Validate pair group names
        for (a, b, _) in &pair_vec {
            if !self.data.group_indices.contains_key(a) {
                return Err(PyValueError::new_err(format!("Unknown group: '{}'", a)));
            }
            if !self.data.group_indices.contains_key(b) {
                return Err(PyValueError::new_err(format!("Unknown group: '{}'", b)));
            }
        }

        // Parse transforms and determine batch size
        let mut transform_arrays: HashMap<String, Vec<[[f64; 4]; 4]>> = HashMap::new();
        let mut batch_size: Option<usize> = None;

        for dynamic_name in &self.data.dynamic_group_names {
            let arr_obj = transforms.get_item(dynamic_name)?;
            if arr_obj.is_none() {
                return Err(PyValueError::new_err(format!(
                    "Missing transform for dynamic group: '{}'",
                    dynamic_name
                )));
            }
            let arr_obj = arr_obj.unwrap();

            let arr_any = arr_obj;

            if let Ok(arr4) = arr_any.extract::<PyReadonlyArray3<f64>>() {
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
                let mut tfs = Vec::with_capacity(n);
                for i in 0..n {
                    let base = i * 16;
                    tfs.push([
                        [slice[base], slice[base+1], slice[base+2], slice[base+3]],
                        [slice[base+4], slice[base+5], slice[base+6], slice[base+7]],
                        [slice[base+8], slice[base+9], slice[base+10], slice[base+11]],
                        [slice[base+12], slice[base+13], slice[base+14], slice[base+15]],
                    ]);
                }
                transform_arrays.insert(dynamic_name.clone(), tfs);
            } else if let Ok(arr2) = arr_any.extract::<PyReadonlyArray2<f64>>() {
                let tf = extract_transform_4x4(&arr2)?;
                if batch_size.is_none() {
                    batch_size = Some(1);
                } else if batch_size != Some(1) {
                    return Err(PyValueError::new_err(
                        "Mixing single and batch transforms"
                    ));
                }
                transform_arrays.insert(dynamic_name.clone(), vec![tf]);
            } else {
                return Err(PyValueError::new_err(format!(
                    "Transform for '{}' must be numpy array",
                    dynamic_name
                )));
            }
        }

        let n = batch_size.unwrap_or(1);

        // Convert pairs to indices for faster access
        let pair_indices: Vec<(usize, usize, f64)> = pair_vec
            .iter()
            .map(|(a, b, min_dist)| {
                (self.data.group_indices[a], self.data.group_indices[b], *min_dist)
            })
            .collect();

        // Prepare group data
        struct GroupCheckData {
            shape: SharedShape,
            is_static: bool,
            static_isometry: Option<Isometry3<f64>>,
        }

        let group_data: Vec<GroupCheckData> = self.data.groups
            .iter()
            .zip(self.shapes.iter())
            .map(|(g, s)| GroupCheckData {
                shape: s.0.clone(),
                is_static: g.is_static,
                static_isometry: g.get_static_isometry(),
            })
            .collect();

        // Use find_any for early exit - returns first collision found by any thread
        let result: Option<usize> = (0..n)
            .into_par_iter()
            .find_any(|&pose_idx| {
                // Build isometries for this pose
                let mut isometries: Vec<Option<Isometry3<f64>>> = vec![None; self.data.groups.len()];

                for (name, tfs) in &transform_arrays {
                    let group_idx = self.data.group_indices[name];
                    isometries[group_idx] = Some(matrix4_to_isometry(&tfs[pose_idx]));
                }

                // Set static isometries
                for (idx, gd) in group_data.iter().enumerate() {
                    if gd.is_static {
                        isometries[idx] = gd.static_isometry.clone();
                    }
                }

                // Check if any pair collides
                pair_indices.iter().any(|&(idx_a, idx_b, min_dist)| {
                    let iso_a = match &isometries[idx_a] {
                        Some(iso) => iso,
                        None => return false,
                    };
                    let iso_b = match &isometries[idx_b] {
                        Some(iso) => iso,
                        None => return false,
                    };

                    let shape_a = &group_data[idx_a].shape;
                    let shape_b = &group_data[idx_b].shape;

                    if min_dist > 0.0 {
                        let dist = query::distance(iso_a, shape_a.as_ref(), iso_b, shape_b.as_ref())
                            .unwrap_or(f64::MAX);
                        dist < min_dist
                    } else {
                        query::intersection_test(iso_a, shape_a.as_ref(), iso_b, shape_b.as_ref())
                            .unwrap_or(false)
                    }
                })
            });

        Ok(result)
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

// ============================================================================
// Helper Functions
// ============================================================================

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
/// Must be called BEFORE the first parallel operation (check/check_any).
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
