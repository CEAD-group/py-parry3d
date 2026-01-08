"""
Test to investigate ConvexHull vs ConvexHull-as-TriMesh collision discrepancy.

The benchmark showed different collision counts:
- ConvexHull: 19,774 collisions
- ConvexHull-as-TriMesh: 13,533 collisions

ROOT CAUSE FOUND:
- ConvexHull (ConvexPolyhedron in parry3d) is treated as SOLID
- TriMesh is treated as HOLLOW (surface only)

When a sphere is completely INSIDE a ConvexHull, it collides.
When a sphere is completely INSIDE a TriMesh, it does NOT collide
(because TriMesh only detects surface intersections).
"""

import numpy as np
import pytest
import trimesh
from pathlib import Path

import py_parry3d as pp

TEST_ASSETS = Path(__file__).parent / "test_assets"


def create_complex_mesh():
    """Create a complex procedural mesh (Stanford bunny-like shape)."""
    # Use a sphere with some noise to simulate a complex organic shape
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.1)

    # Add some noise to make it non-convex
    noise = np.random.RandomState(42).randn(len(mesh.vertices), 3) * 0.02
    mesh.vertices += noise

    return mesh


def load_utah_teapot():
    """Load the Utah teapot mesh if available, otherwise create a substitute."""
    teapot_path = TEST_ASSETS / "utah_teapot.obj"
    if teapot_path.exists():
        mesh = trimesh.load(teapot_path, force='mesh')
        # Scale to reasonable size (approx 1 unit extent)
        mesh.apply_scale(0.15)
        return mesh
    else:
        # Fallback: create a complex procedural mesh
        return create_complex_mesh()


def create_simple_box():
    """Create a simple box mesh."""
    return trimesh.creation.box(extents=[0.2, 0.15, 0.1])


def create_icosphere():
    """Create a smooth sphere-like mesh."""
    return trimesh.creation.icosphere(subdivisions=2, radius=0.1)


def generate_random_transforms(n: int, seed: int = 42) -> np.ndarray:
    """Generate random transforms in a grid pattern with noise."""
    rng = np.random.RandomState(seed)
    transforms = np.zeros((n, 4, 4), dtype=np.float64)

    # Generate positions in a grid pattern with some noise
    side = int(np.ceil(n ** (1/3)))
    idx = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if idx >= n:
                    break
                tf = np.eye(4)
                tf[:3, 3] = [
                    ix * 0.3 + rng.randn() * 0.05,
                    iy * 0.3 + rng.randn() * 0.05,
                    iz * 0.3 + rng.randn() * 0.05,
                ]
                # Add random rotation
                angles = rng.randn(3) * 0.5
                rot = trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2])
                tf[:3, :3] = rot[:3, :3]
                transforms[idx] = tf
                idx += 1
            if idx >= n:
                break
        if idx >= n:
            break

    return transforms[:n]


class TestConvexHullConsistency:
    """Test that ConvexHull and ConvexHull-as-TriMesh give consistent results."""

    @pytest.fixture
    def complex_mesh(self):
        return create_complex_mesh()

    @pytest.fixture
    def simple_mesh(self):
        return create_simple_box()

    def test_hull_geometry_extraction(self, simple_mesh):
        """Verify that ConvexHull geometry can be extracted correctly."""
        hull = pp.ConvexHull_from_trimesh(simple_mesh)

        # Check that we can access vertices and faces
        vertices = hull.vertices
        faces = hull.faces

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert faces.ndim == 2
        assert faces.shape[1] == 3

        # Verify dtype
        assert vertices.dtype == np.float64
        assert faces.dtype == np.uint32

        # Faces should reference valid vertices
        assert faces.max() < len(vertices)

        print(f"\nSimple box hull: {len(vertices)} vertices, {len(faces)} faces")

    def test_hull_geometry_vs_trimesh_hull(self, complex_mesh):
        """Compare parry3d hull geometry to trimesh hull."""
        # Get parry3d convex hull
        parry_hull = pp.ConvexHull_from_trimesh(complex_mesh)
        parry_verts = parry_hull.vertices
        parry_faces = parry_hull.faces

        # Get trimesh convex hull
        trimesh_hull = complex_mesh.convex_hull
        trimesh_verts = trimesh_hull.vertices
        trimesh_faces = trimesh_hull.faces

        print("\nComplex mesh:")
        print(f"  Original: {len(complex_mesh.vertices)} vertices, {len(complex_mesh.faces)} faces")
        print(f"  Parry hull: {len(parry_verts)} vertices, {len(parry_faces)} faces")
        print(f"  Trimesh hull: {len(trimesh_verts)} vertices, {len(trimesh_faces)} faces")

        # They may not be exactly the same due to different algorithms,
        # but should be similar in size
        assert abs(len(parry_verts) - len(trimesh_verts)) < 50, \
            f"Large discrepancy: parry={len(parry_verts)} vs trimesh={len(trimesh_verts)}"

    def test_single_collision_convexhull_vs_trimesh(self, simple_mesh):
        """Test single collision: ConvexHull vs same geometry as TriMesh."""
        # Create ConvexHull shape
        convex_hull = pp.ConvexHull_from_trimesh(simple_mesh)

        # Create TriMesh from the ConvexHull geometry
        hull_mesh = trimesh.Trimesh(
            vertices=convex_hull.vertices,
            faces=convex_hull.faces
        )
        tri_mesh = pp.TriMesh_from_trimesh(hull_mesh)

        # Create collision groups
        g_hull = pp.CollisionGroup("hull", [convex_hull])
        g_trimesh = pp.CollisionGroup("trimesh", [tri_mesh])

        # Create a sphere to collide against
        sphere_tf = np.eye(4, dtype=np.float64)
        sphere_tf[:3, 3] = [0.15, 0, 0]  # Close to mesh
        g_sphere = pp.CollisionGroup("sphere", [pp.Sphere(0.05)], static=True, transform=sphere_tf)

        # Create worlds
        world_hull = pp.CollisionWorld([g_hull, g_sphere])
        world_trimesh = pp.CollisionWorld([g_trimesh, g_sphere])

        # Test at multiple positions
        n_tests = 100
        transforms = generate_random_transforms(n_tests)

        # Center the transforms near the sphere
        transforms[:, :3, 3] *= 0.5
        transforms[:, :3, 3] += [0.15, 0, 0]

        pairs = [("hull", "sphere", 0.0)]
        pairs_trimesh = [("trimesh", "sphere", 0.0)]

        result_hull = world_hull.check({"hull": transforms}, pairs)
        result_trimesh = world_trimesh.check({"trimesh": transforms}, pairs_trimesh)

        collisions_hull = np.sum(result_hull)
        collisions_trimesh = np.sum(result_trimesh)

        print("\nSingle collision test (100 poses):")
        print(f"  ConvexHull collisions: {collisions_hull}")
        print(f"  TriMesh collisions: {collisions_trimesh}")

        # They should be identical!
        assert collisions_hull == collisions_trimesh, \
            f"Collision count mismatch: ConvexHull={collisions_hull}, TriMesh={collisions_trimesh}"

    def test_batch_collision_consistency(self, complex_mesh):
        """Test batch collision: ConvexHull vs same geometry as TriMesh."""
        # Create ConvexHull shape
        convex_hull = pp.ConvexHull_from_trimesh(complex_mesh)

        # Create TriMesh from the ConvexHull geometry
        hull_mesh = trimesh.Trimesh(
            vertices=convex_hull.vertices,
            faces=convex_hull.faces
        )
        tri_mesh = pp.TriMesh_from_trimesh(hull_mesh)

        # Create collision groups
        g_hull = pp.CollisionGroup("mesh", [convex_hull])
        g_trimesh = pp.CollisionGroup("mesh", [tri_mesh])

        # Create multiple spheres as obstacles
        sphere_groups = []
        n_spheres = 50
        rng = np.random.RandomState(123)
        for i in range(n_spheres):
            tf = np.eye(4)
            tf[:3, 3] = rng.randn(3) * 0.3
            sphere_groups.append(
                pp.CollisionGroup(f"sphere_{i}", [pp.Sphere(0.03)], static=True, transform=tf)
            )

        # Create worlds
        world_hull = pp.CollisionWorld([g_hull] + sphere_groups)
        world_trimesh = pp.CollisionWorld([g_trimesh] + sphere_groups)

        # Generate test poses
        n_poses = 500
        transforms = generate_random_transforms(n_poses, seed=456)
        transforms[:, :3, 3] *= 1.0  # Scale positions

        # Define pairs
        sphere_names = [f"sphere_{i}" for i in range(n_spheres)]
        pairs = pp.pairs_vs(["mesh"], sphere_names, min_distance=0.0)

        # Run collision checks
        result_hull = world_hull.check({"mesh": transforms}, pairs)
        result_trimesh = world_trimesh.check({"mesh": transforms}, pairs)

        total_hull = np.sum(result_hull)
        total_trimesh = np.sum(result_trimesh)

        print(f"\nBatch collision test ({n_poses} poses x {n_spheres} spheres):")
        print(f"  ConvexHull collisions: {total_hull}")
        print(f"  TriMesh collisions: {total_trimesh}")

        # Compare per-pose results
        per_pose_hull = result_hull.sum(axis=1)
        per_pose_trimesh = result_trimesh.sum(axis=1)

        mismatches = np.sum(per_pose_hull != per_pose_trimesh)
        print(f"  Poses with different collision count: {mismatches}/{n_poses}")

        # NOTE: They will NOT be identical because:
        # - ConvexHull (solid) detects spheres INSIDE it
        # - TriMesh (hollow) only detects surface intersections
        # This test now documents this expected behavior
        assert total_hull >= total_trimesh, \
            "ConvexHull (solid) should detect at least as many collisions as TriMesh (hollow)"
        if total_hull > total_trimesh:
            print(f"  NOTE: ConvexHull detected more collisions ({total_hull} vs {total_trimesh})")
            print("        because it is SOLID, detecting spheres fully inside it.")

    def test_exact_same_shape_collision(self):
        """Test collisions using exact same shape object."""
        # Create a simple box
        mesh = create_simple_box()

        # Create ConvexHull
        convex_hull = pp.ConvexHull_from_trimesh(mesh)

        # Get the hull geometry
        hull_verts = convex_hull.vertices.copy()
        hull_faces = convex_hull.faces.copy()

        print(f"\nHull geometry: {len(hull_verts)} vertices, {len(hull_faces)} faces")

        # Create TriMesh from exact same geometry
        tri_mesh = pp.TriMesh(hull_verts, hull_faces)

        # Create test sphere
        sphere_tf = np.eye(4)
        sphere_tf[:3, 3] = [0.0, 0.0, 0.0]
        sphere = pp.CollisionGroup("sphere", [pp.Sphere(0.08)], static=True, transform=sphere_tf)

        # Create groups with different shape types
        g_hull = pp.CollisionGroup("test", [convex_hull])
        g_trimesh = pp.CollisionGroup("test", [tri_mesh])

        world_hull = pp.CollisionWorld([g_hull, sphere])
        world_trimesh = pp.CollisionWorld([g_trimesh, sphere])

        # Test at specific positions
        test_positions = [
            [0.0, 0.0, 0.0],       # Origin (overlapping)
            [0.05, 0.0, 0.0],      # Slight offset
            [0.1, 0.0, 0.0],       # Edge
            [0.15, 0.0, 0.0],      # Just touching
            [0.2, 0.0, 0.0],       # Gap
            [0.0, 0.05, 0.0],
            [0.0, 0.1, 0.0],
            [0.05, 0.05, 0.05],    # Diagonal
        ]

        pairs = [("test", "sphere", 0.0)]

        print("\nPosition-by-position comparison:")
        mismatches = []
        for pos in test_positions:
            tf = np.eye(4, dtype=np.float64)
            tf[:3, 3] = pos

            result_hull = world_hull.check({"test": tf}, pairs)[0]
            result_trimesh = world_trimesh.check({"test": tf}, pairs)[0]

            match = "OK" if result_hull == result_trimesh else "MISMATCH"
            if result_hull != result_trimesh:
                mismatches.append(pos)
            print(f"  pos={pos}: hull={result_hull}, trimesh={result_trimesh} [{match}]")

        assert len(mismatches) == 0, f"Mismatches at positions: {mismatches}"


class TestConvexHullCollisionAlgorithm:
    """Test to understand how ConvexHull collision works vs TriMesh."""

    def test_convex_hull_is_solid(self):
        """Test that ConvexHull is treated as a solid shape."""
        # Create a box
        mesh = create_simple_box()
        hull = pp.ConvexHull_from_trimesh(mesh)

        # Create a small sphere at origin
        sphere_tf = np.eye(4)
        sphere = pp.CollisionGroup("sphere", [pp.Sphere(0.01)], static=True, transform=sphere_tf)

        # Box at origin should collide with sphere at origin (inside the box)
        g = pp.CollisionGroup("box", [hull])
        world = pp.CollisionWorld([g, sphere])

        tf = np.eye(4, dtype=np.float64)
        result = world.check({"box": tf}, [("box", "sphere", 0.0)])

        print(f"\nConvexHull at origin with sphere at origin: collides={result[0]}")
        assert result[0], "ConvexHull should be solid - sphere inside should collide"

    def test_trimesh_is_solid_or_hollow(self):
        """Test if TriMesh is treated as solid or hollow."""
        # Create a box
        mesh = create_simple_box()
        hull = pp.ConvexHull_from_trimesh(mesh)
        hull_mesh = trimesh.Trimesh(vertices=hull.vertices, faces=hull.faces)
        tri = pp.TriMesh_from_trimesh(hull_mesh)

        # Create a small sphere at origin
        sphere_tf = np.eye(4)
        sphere = pp.CollisionGroup("sphere", [pp.Sphere(0.01)], static=True, transform=sphere_tf)

        # Box at origin - sphere inside
        g = pp.CollisionGroup("box", [tri])
        world = pp.CollisionWorld([g, sphere])

        tf = np.eye(4, dtype=np.float64)
        result = world.check({"box": tf}, [("box", "sphere", 0.0)])

        print(f"\nTriMesh at origin with sphere at origin: collides={result[0]}")
        # Note: This reveals whether TriMesh is solid or hollow

    def test_edge_collision_comparison(self):
        """Test collision detection at mesh edges."""
        mesh = create_simple_box()
        hull = pp.ConvexHull_from_trimesh(mesh)
        hull_mesh = trimesh.Trimesh(vertices=hull.vertices, faces=hull.faces)
        tri = pp.TriMesh_from_trimesh(hull_mesh)

        # Test sphere at various distances from the surface
        distances = [0.0, 0.05, 0.08, 0.10, 0.11, 0.12, 0.15]
        sphere_radius = 0.02

        print(f"\nEdge collision test (box half-extent 0.1, sphere radius {sphere_radius}):")

        for dist in distances:
            sphere_tf = np.eye(4)
            sphere_tf[:3, 3] = [dist, 0, 0]

            sphere_hull = pp.CollisionGroup("s", [pp.Sphere(sphere_radius)], static=True, transform=sphere_tf)
            sphere_tri = pp.CollisionGroup("s", [pp.Sphere(sphere_radius)], static=True, transform=sphere_tf)

            g_hull = pp.CollisionGroup("m", [hull])
            g_tri = pp.CollisionGroup("m", [tri])

            world_hull = pp.CollisionWorld([g_hull, sphere_hull])
            world_tri = pp.CollisionWorld([g_tri, sphere_tri])

            tf = np.eye(4, dtype=np.float64)

            result_hull = world_hull.check({"m": tf}, [("m", "s", 0.0)])[0]
            result_tri = world_tri.check({"m": tf}, [("m", "s", 0.0)])[0]

            match = "OK" if result_hull == result_tri else "MISMATCH"
            print(f"  dist={dist:.2f}: hull={result_hull}, tri={result_tri} [{match}]")


class TestRealMeshCollision:
    """Test collision detection with real complex meshes (Utah Teapot)."""

    @pytest.fixture
    def teapot_mesh(self):
        return load_utah_teapot()

    def test_teapot_convexhull_vs_trimesh(self, teapot_mesh):
        """
        Comprehensive test demonstrating solid vs hollow difference.

        This test shows why ConvexHull reports more collisions than
        ConvexHull-as-TriMesh: ConvexHull detects spheres INSIDE it,
        while TriMesh only detects surface intersections.
        """
        print(f"\n{'='*70}")
        print("Utah Teapot: ConvexHull (solid) vs TriMesh (hollow)")
        print(f"{'='*70}")

        # Create ConvexHull shape
        convex_hull = pp.ConvexHull_from_trimesh(teapot_mesh)
        hull_verts = convex_hull.vertices
        hull_faces = convex_hull.faces

        print("\nTeapot stats:")
        print(f"  Original mesh: {len(teapot_mesh.vertices):,} vertices, {len(teapot_mesh.faces):,} faces")
        print(f"  Convex hull: {len(hull_verts):,} vertices, {len(hull_faces):,} faces")

        # Create TriMesh from hull geometry
        hull_mesh = trimesh.Trimesh(vertices=hull_verts, faces=hull_faces)
        tri_mesh = pp.TriMesh_from_trimesh(hull_mesh)

        # Create collision groups for the mesh
        g_hull = pp.CollisionGroup("mesh", [convex_hull])
        g_trimesh = pp.CollisionGroup("mesh", [tri_mesh])

        # Create grid of small spheres around and inside the mesh
        mesh_center = teapot_mesh.centroid
        mesh_extent = np.max(teapot_mesh.extents)

        sphere_groups = []
        n_spheres = 0
        sphere_radius = mesh_extent * 0.03
        grid_size = 8  # 8x8x8 = 512 spheres
        spacing = mesh_extent * 1.5 / grid_size

        rng = np.random.RandomState(999)
        for ix in range(grid_size):
            for iy in range(grid_size):
                for iz in range(grid_size):
                    x = mesh_center[0] + (ix - grid_size/2) * spacing
                    y = mesh_center[1] + (iy - grid_size/2) * spacing
                    z = mesh_center[2] + (iz - grid_size/2) * spacing
                    # Add small random offset
                    x += rng.randn() * spacing * 0.1
                    y += rng.randn() * spacing * 0.1
                    z += rng.randn() * spacing * 0.1

                    tf = np.eye(4)
                    tf[:3, 3] = [x, y, z]
                    sphere_groups.append(
                        pp.CollisionGroup(f"sphere_{n_spheres}", [pp.Sphere(sphere_radius)],
                                          static=True, transform=tf)
                    )
                    n_spheres += 1

        print(f"  Spheres: {n_spheres} (radius={sphere_radius:.4f})")

        # Create worlds
        world_hull = pp.CollisionWorld([g_hull] + sphere_groups)
        world_trimesh = pp.CollisionWorld([g_trimesh] + sphere_groups)

        # Generate random poses for the mesh
        n_poses = 200
        transforms = np.zeros((n_poses, 4, 4), dtype=np.float64)
        rng = np.random.RandomState(123)

        for i in range(n_poses):
            tf = np.eye(4)
            # Random rotation
            angles = rng.randn(3) * 0.3
            rot = trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2])
            tf[:3, :3] = rot[:3, :3]
            # Random translation near center
            tf[:3, 3] = mesh_center + rng.randn(3) * mesh_extent * 0.3
            transforms[i] = tf

        # Define pairs
        sphere_names = [f"sphere_{i}" for i in range(n_spheres)]
        pairs = pp.pairs_vs(["mesh"], sphere_names, min_distance=0.0)

        print(f"  Poses: {n_poses}")
        print(f"  Total collision checks: {n_poses * len(pairs):,}")

        # Run collision checks
        result_hull = world_hull.check({"mesh": transforms}, pairs)
        result_trimesh = world_trimesh.check({"mesh": transforms}, pairs)

        total_hull = int(np.sum(result_hull))
        total_trimesh = int(np.sum(result_trimesh))

        print("\nResults:")
        print(f"  ConvexHull (solid) collisions:  {total_hull:,}")
        print(f"  TriMesh (hollow) collisions:    {total_trimesh:,}")
        print(f"  Difference:                     {total_hull - total_trimesh:,}")
        print(f"  Ratio:                          {total_hull / max(total_trimesh, 1):.2f}x")

        # The key insight: ConvexHull should have MORE collisions
        # because it detects spheres that are fully INSIDE
        print(f"\n{'='*70}")
        print("EXPLANATION:")
        print("  ConvexHull is SOLID - detects objects inside AND touching surface")
        print("  TriMesh is HOLLOW - only detects objects touching the surface")
        print("  Extra collisions from ConvexHull = spheres fully inside the hull")
        print(f"{'='*70}")

        assert total_hull >= total_trimesh, \
            "ConvexHull should detect at least as many collisions as TriMesh"

    def test_inside_vs_surface_collision(self, teapot_mesh):
        """
        Explicitly test the difference between inside and surface collision.
        """
        print(f"\n{'='*70}")
        print("Inside vs Surface Collision Test")
        print(f"{'='*70}")

        # Create ConvexHull
        convex_hull = pp.ConvexHull_from_trimesh(teapot_mesh)
        hull_verts = convex_hull.vertices
        hull_mesh = trimesh.Trimesh(vertices=hull_verts, faces=convex_hull.faces)
        tri_mesh = pp.TriMesh_from_trimesh(hull_mesh)

        mesh_center = hull_mesh.centroid
        mesh_extent = np.max(hull_mesh.extents)

        # Test 1: Sphere at center of mesh (definitely inside)
        sphere_inside_tf = np.eye(4)
        sphere_inside_tf[:3, 3] = mesh_center
        sphere_inside = pp.CollisionGroup("s", [pp.Sphere(0.001)], static=True, transform=sphere_inside_tf)

        g_hull = pp.CollisionGroup("m", [convex_hull])
        g_trimesh = pp.CollisionGroup("m", [tri_mesh])

        world_hull_inside = pp.CollisionWorld([g_hull, sphere_inside])
        world_trimesh_inside = pp.CollisionWorld([g_trimesh, sphere_inside])

        tf = np.eye(4, dtype=np.float64)
        hull_inside = world_hull_inside.check({"m": tf}, [("m", "s", 0.0)])[0]
        tri_inside = world_trimesh_inside.check({"m": tf}, [("m", "s", 0.0)])[0]

        print("\nSphere at mesh CENTER (definitely inside):")
        print(f"  ConvexHull detects: {hull_inside}")
        print(f"  TriMesh detects:    {tri_inside}")

        # Test 2: Sphere far outside mesh
        sphere_outside_tf = np.eye(4)
        sphere_outside_tf[:3, 3] = mesh_center + [mesh_extent * 2, 0, 0]
        sphere_outside = pp.CollisionGroup("s", [pp.Sphere(0.001)], static=True, transform=sphere_outside_tf)

        world_hull_outside = pp.CollisionWorld([g_hull, sphere_outside])
        world_trimesh_outside = pp.CollisionWorld([g_trimesh, sphere_outside])

        hull_outside = world_hull_outside.check({"m": tf}, [("m", "s", 0.0)])[0]
        tri_outside = world_trimesh_outside.check({"m": tf}, [("m", "s", 0.0)])[0]

        print("\nSphere FAR OUTSIDE mesh:")
        print(f"  ConvexHull detects: {hull_outside}")
        print(f"  TriMesh detects:    {tri_outside}")

        # Test 3: Sphere at surface (touching)
        # Find a point on the surface
        surface_point = hull_mesh.vertices[0]
        sphere_surface_tf = np.eye(4)
        sphere_surface_tf[:3, 3] = surface_point
        sphere_surface = pp.CollisionGroup("s", [pp.Sphere(0.01)], static=True, transform=sphere_surface_tf)

        world_hull_surface = pp.CollisionWorld([g_hull, sphere_surface])
        world_trimesh_surface = pp.CollisionWorld([g_trimesh, sphere_surface])

        hull_surface = world_hull_surface.check({"m": tf}, [("m", "s", 0.0)])[0]
        tri_surface = world_trimesh_surface.check({"m": tf}, [("m", "s", 0.0)])[0]

        print("\nSphere at mesh SURFACE (touching):")
        print(f"  ConvexHull detects: {hull_surface}")
        print(f"  TriMesh detects:    {tri_surface}")

        print(f"\n{'='*70}")
        print("CONCLUSION:")
        print("  - INSIDE: ConvexHull=True, TriMesh=False (key difference!)")
        print("  - OUTSIDE: Both=False (both agree)")
        print("  - SURFACE: Both=True (both agree)")
        print(f"{'='*70}")

        # Assertions
        assert hull_inside, "ConvexHull should detect sphere inside"
        assert not tri_inside, "TriMesh should NOT detect sphere inside (hollow)"
        assert not hull_outside, "ConvexHull should NOT detect sphere far outside"
        assert not tri_outside, "TriMesh should NOT detect sphere far outside"
        assert hull_surface, "ConvexHull should detect sphere at surface"
        assert tri_surface, "TriMesh should detect sphere at surface"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
