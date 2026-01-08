"""Tests for py_parry3d collision detection."""

import numpy as np
import pytest

import py_parry3d as pp


class TestShapes:
    """Test shape creation."""

    def test_box(self):
        box = pp.Box([0.5, 0.3, 0.2])
        assert "Box" in repr(box)

    def test_sphere(self):
        sphere = pp.Sphere(0.1)
        assert "Sphere" in repr(sphere)

    def test_capsule(self):
        capsule = pp.Capsule(0.5, 0.1)
        assert "Capsule" in repr(capsule)

    def test_cylinder(self):
        cylinder = pp.Cylinder(0.5, 0.1)
        assert "Cylinder" in repr(cylinder)

    def test_trimesh(self):
        # Simple tetrahedron
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ], dtype=np.uint32)
        mesh = pp.TriMesh(vertices, faces)
        assert "TriMesh" in repr(mesh)

    def test_convex_hull_from_trimesh(self):
        import trimesh

        # Create a simple mesh (cube)
        mesh = trimesh.creation.box()

        # Create convex hull
        hull = pp.ConvexHull_from_trimesh(mesh)

        # Check geometry is accessible
        assert hull.vertices.shape[1] == 3
        assert hull.faces.shape[1] == 3
        assert hull.vertices.dtype == np.float64
        assert hull.faces.dtype == np.uint32
        assert "ConvexHull" in repr(hull)

        # Use in collision group
        group = pp.CollisionGroup("test", [hull])
        static_tf = np.eye(4, dtype=np.float64)
        static_tf[:3, 3] = [0, 0, -10]
        floor = pp.CollisionGroup("floor", [pp.Box([10, 10, 0.1])], static=True, transform=static_tf)
        world = pp.CollisionWorld([group, floor])

        # Should work in collision check
        tf = np.eye(4, dtype=np.float64)
        result = world.check({"test": tf}, [("test", "floor", 0.0)])
        assert result is not None
        assert not result[0]  # No collision - floor is far below


class TestCollisionObject:
    """Test CollisionObject creation."""

    def test_without_transform(self):
        obj = pp.CollisionObject(pp.Box([0.5, 0.3, 0.2]))
        assert obj is not None

    def test_with_transform(self):
        tf = np.eye(4, dtype=np.float64)
        tf[:3, 3] = [1, 2, 3]
        obj = pp.CollisionObject(pp.Sphere(0.1), transform=tf)
        assert obj is not None


class TestCollisionGroup:
    """Test CollisionGroup creation."""

    def test_dynamic_group(self):
        group = pp.CollisionGroup("test", [pp.Box([0.1, 0.1, 0.1])])
        assert group.name == "test"
        assert not group.is_static
        assert len(group) == 1

    def test_static_group(self):
        tf = np.eye(4, dtype=np.float64)
        group = pp.CollisionGroup(
            "static",
            [pp.Box([1, 1, 0.1])],
            static=True,
            transform=tf,
        )
        assert group.name == "static"
        assert group.is_static

    def test_static_group_requires_transform(self):
        with pytest.raises(ValueError):
            pp.CollisionGroup("static", [pp.Box([1, 1, 0.1])], static=True)

    def test_multiple_objects(self):
        group = pp.CollisionGroup("multi", [
            pp.Box([0.1, 0.1, 0.1]),
            pp.Sphere(0.05),
            pp.Capsule(0.2, 0.05),
        ])
        assert len(group) == 3


class TestCollisionWorld:
    """Test CollisionWorld creation and checking."""

    @pytest.fixture
    def simple_world(self):
        """Create a simple world with one dynamic and one static group."""
        g1 = pp.CollisionGroup("robot", [pp.Box([0.2, 0.2, 0.2])])

        static_tf = np.eye(4, dtype=np.float64)
        static_tf[:3, 3] = [0, 0, -0.5]
        g2 = pp.CollisionGroup(
            "floor",
            [pp.Box([10, 10, 0.1])],
            static=True,
            transform=static_tf,
        )

        return pp.CollisionWorld([g1, g2])

    def test_world_creation(self, simple_world):
        assert len(simple_world) == 2
        assert simple_world.dynamic_groups == ["robot"]
        assert simple_world.static_groups == ["floor"]

    def test_single_check_no_collision(self, simple_world):
        # Robot at origin, floor below - no collision
        tf = np.eye(4, dtype=np.float64)
        tf[:3, 3] = [0, 0, 1]  # Robot 1m above origin

        result = simple_world.check(
            {"robot": tf},
            [("robot", "floor", 0.0)],
        )
        assert result.shape == (1,)
        assert not result[0]

    def test_single_check_collision(self, simple_world):
        # Robot at floor level - collision
        tf = np.eye(4, dtype=np.float64)
        tf[:3, 3] = [0, 0, -0.4]  # Robot at z=-0.4, floor at z=-0.5

        result = simple_world.check(
            {"robot": tf},
            [("robot", "floor", 0.0)],
        )
        assert result.shape == (1,)
        assert result[0]

    def test_batch_check(self, simple_world):
        N = 100
        transforms = np.tile(np.eye(4), (N, 1, 1)).astype(np.float64)

        # Half above floor, half colliding
        transforms[:50, 2, 3] = 1.0  # z = 1 (no collision)
        transforms[50:, 2, 3] = -0.4  # z = -0.4 (collision)

        result = simple_world.check(
            {"robot": transforms},
            [("robot", "floor", 0.0)],
        )
        assert result.shape == (N, 1)
        assert result[:50, 0].sum() == 0  # No collisions in first half
        assert result[50:, 0].sum() == 50  # All collisions in second half


class TestHelpers:
    """Test helper functions."""

    def test_all_pairs(self):
        names = ["a", "b", "c", "d"]
        pairs = pp.all_pairs(names)
        expected = [("a", "b", 0.0), ("a", "c", 0.0), ("a", "d", 0.0), ("b", "c", 0.0), ("b", "d", 0.0), ("c", "d", 0.0)]
        assert pairs == expected

    def test_all_pairs_skip_adjacent(self):
        names = ["a", "b", "c", "d"]
        pairs = pp.all_pairs(names, skip_adjacent=1)
        expected = [("a", "c", 0.0), ("a", "d", 0.0), ("b", "d", 0.0)]
        assert pairs == expected

    def test_pairs_vs_single(self):
        names = ["a", "b", "c"]
        pairs = pp.pairs_vs(names, "env")
        expected = [("a", "env", 0.0), ("b", "env", 0.0), ("c", "env", 0.0)]
        assert pairs == expected

    def test_pairs_vs_list(self):
        names = ["a", "b"]
        pairs = pp.pairs_vs(names, ["env1", "env2"])
        expected = [("a", "env1", 0.0), ("a", "env2", 0.0), ("b", "env1", 0.0), ("b", "env2", 0.0)]
        assert pairs == expected


class TestSerialization:
    """Test serialization."""

    def test_to_bytes_from_bytes(self):
        g1 = pp.CollisionGroup("test", [pp.Box([0.1, 0.1, 0.1])])
        static_tf = np.eye(4, dtype=np.float64)
        g2 = pp.CollisionGroup("static", [pp.Sphere(0.5)], static=True, transform=static_tf)

        world = pp.CollisionWorld([g1, g2])
        data = world.to_bytes()
        assert len(data) > 0

        world2 = pp.CollisionWorld.from_bytes(data)
        assert len(world2) == 2
        assert world2.dynamic_groups == ["test"]
        assert world2.static_groups == ["static"]

    def test_pickle(self):
        import pickle

        g1 = pp.CollisionGroup("test", [pp.Box([0.1, 0.1, 0.1])])
        static_tf = np.eye(4, dtype=np.float64)
        g2 = pp.CollisionGroup("static", [pp.Sphere(0.5)], static=True, transform=static_tf)

        world = pp.CollisionWorld([g1, g2])
        data = pickle.dumps(world)
        world2 = pickle.loads(data)

        assert len(world2) == 2


class TestMinDistance:
    """Test minimum distance threshold."""

    def test_min_distance_no_collision(self):
        """Objects far apart with no min_distance - no collision."""
        g1 = pp.CollisionGroup("a", [pp.Sphere(0.1)])
        g2 = pp.CollisionGroup("b", [pp.Sphere(0.1)])
        world = pp.CollisionWorld([g1, g2])

        tf_a = np.eye(4, dtype=np.float64)
        tf_b = np.eye(4, dtype=np.float64)
        tf_b[0, 3] = 0.5  # 0.5m apart, spheres radius 0.1 each -> gap of 0.3m

        result = world.check({"a": tf_a, "b": tf_b}, [("a", "b", 0.0)])
        assert not result[0]

    def test_min_distance_triggers_collision(self):
        """Objects far apart but within min_distance - collision."""
        g1 = pp.CollisionGroup("a", [pp.Sphere(0.1)])
        g2 = pp.CollisionGroup("b", [pp.Sphere(0.1)])
        world = pp.CollisionWorld([g1, g2])

        tf_a = np.eye(4, dtype=np.float64)
        tf_b = np.eye(4, dtype=np.float64)
        tf_b[0, 3] = 0.5  # 0.5m apart, gap of 0.3m

        # With min_distance=0.4, should trigger (gap 0.3 < 0.4)
        result = world.check({"a": tf_a, "b": tf_b}, [("a", "b", 0.4)])
        assert result[0]

    def test_pairs_with_min_distance(self):
        """Test helper functions with min_distance."""
        pairs = pp.all_pairs(["a", "b", "c"], min_distance=0.05)
        assert all(len(p) == 3 for p in pairs)
        assert all(p[2] == 0.05 for p in pairs)

        pairs = pp.pairs_vs(["a", "b"], "env", min_distance=0.1)
        assert all(len(p) == 3 for p in pairs)
        assert all(p[2] == 0.1 for p in pairs)


class TestCheckAny:
    """Test early-exit collision checking."""

    def test_check_any_no_collision(self):
        """No collisions returns None."""
        g1 = pp.CollisionGroup("a", [pp.Sphere(0.1)])
        tf = np.eye(4, dtype=np.float64)
        g2 = pp.CollisionGroup("b", [pp.Sphere(0.1)], static=True, transform=tf)
        world = pp.CollisionWorld([g1, g2])

        # Move a far from b
        transforms = np.tile(np.eye(4), (10, 1, 1)).astype(np.float64)
        transforms[:, 0, 3] = 10.0  # Far apart

        result = world.check_any({"a": transforms}, [("a", "b", 0.0)])
        assert result is None

    def test_check_any_with_collision(self):
        """Collision returns index."""
        g1 = pp.CollisionGroup("a", [pp.Sphere(0.1)])
        tf = np.eye(4, dtype=np.float64)
        g2 = pp.CollisionGroup("b", [pp.Sphere(0.1)], static=True, transform=tf)
        world = pp.CollisionWorld([g1, g2])

        transforms = np.tile(np.eye(4), (10, 1, 1)).astype(np.float64)
        transforms[:5, 0, 3] = 10.0  # First 5 far apart
        transforms[5:, 0, 3] = 0.0   # Last 5 colliding

        result = world.check_any({"a": transforms}, [("a", "b", 0.0)])
        assert result is not None
        assert result >= 5  # Should be one of the colliding poses


class TestThreading:
    """Test threading configuration."""

    def test_get_set_threads(self):
        n = pp.get_num_threads()
        assert n > 0

        # Note: set_num_threads only works before first parallel operation
        # and may fail silently if thread pool is already initialized
