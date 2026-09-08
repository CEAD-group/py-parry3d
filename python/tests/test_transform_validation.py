"""Transforms handed to py_parry3d must be rigid (issue #12).

Every Python entry point that accepts a 4x4 transform validates it: finite
entries, a ``[0, 0, 0, 1]`` bottom row, and a 3x3 block that is a proper
rotation. Before this was enforced, a non-rigid matrix produced a silently
wrong answer - a 2x scaled box moved but kept its original size, and a NaN
pose reported "collision" - which for a collision checker means reporting
clear when it is not.
"""

import numpy as np
import py_parry3d as pp
import pytest


def rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float64)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


# --- the invalid matrices, each with a fragment its message must mention -----

INVALID = {
    "nan_translation": (
        lambda: _with(np.eye(4), lambda m: m.__setitem__((0, 3), np.nan)),
        "finite",
    ),
    "nan_rotation": (
        lambda: _with(np.eye(4), lambda m: m.__setitem__((1, 1), np.nan)),
        "finite",
    ),
    "inf_translation": (
        lambda: _with(np.eye(4), lambda m: m.__setitem__((2, 3), np.inf)),
        "finite",
    ),
    "neg_inf": (
        lambda: _with(np.eye(4), lambda m: m.__setitem__((0, 0), -np.inf)),
        "finite",
    ),
    "uniform_scale": (
        lambda: _with(np.eye(4), lambda m: m.__setitem__((slice(0, 3), slice(0, 3)), np.eye(3) * 2.0)),
        "orthonormal",
    ),
    "scale_and_translate": (
        lambda: _with(
            np.eye(4),
            lambda m: (
                m.__setitem__((slice(0, 3), slice(0, 3)), np.eye(3) * 2.0),
                m.__setitem__((0, 3), 1.2),
            ),
        ),
        "orthonormal",
    ),
    "shear": (
        lambda: _with(np.eye(4), lambda m: m.__setitem__((0, 1), 0.5)),
        "orthonormal",
    ),
    "degenerate_zero_block": (
        lambda: _with(np.eye(4), lambda m: m.__setitem__((slice(0, 3), slice(0, 3)), np.zeros((3, 3)))),
        "orthonormal",
    ),
    "reflection": (
        # Mirror one axis: orthonormal, but determinant -1.
        lambda: _with(np.eye(4), lambda m: m.__setitem__((2, 2), -1.0)),
        "determinant",
    ),
    "bad_bottom_row": (
        lambda: _with(np.eye(4), lambda m: m.__setitem__(3, [9.0, 9.0, 9.0, 7.0])),
        "bottom row",
    ),
    "perspective_term": (
        lambda: _with(np.eye(4), lambda m: m.__setitem__((3, 0), 0.1)),
        "bottom row",
    ),
    "bottom_right_zero": (
        lambda: _with(np.eye(4), lambda m: m.__setitem__((3, 3), 0.0)),
        "bottom row",
    ),
}


def _with(m, fn):
    m = np.array(m, dtype=np.float64)
    fn(m)
    return m


def invalid_cases():
    return [pytest.param(factory, fragment, id=name) for name, (factory, fragment) in INVALID.items()]


# --- legitimate transforms ---------------------------------------------------


def drifted_rotation(n: int = 50) -> np.ndarray:
    """A rotation built from ~n composed rotations, carrying real float drift.

    This is what an accumulated forward-kinematics product looks like: a
    genuine rotation whose orthonormality holds only to within rounding error.
    It must be accepted.
    """
    m = np.eye(4, dtype=np.float64)
    rng = np.random.default_rng(0xC0FFEE)
    for i in range(n):
        angle = rng.uniform(-np.pi, np.pi)
        axis = (rotation_x, rotation_y, rotation_z)[i % 3]
        m = m @ axis(angle)
    return m


VALID = {
    "identity": lambda: np.eye(4, dtype=np.float64),
    "pure_translation": lambda: _with(np.eye(4), lambda m: m.__setitem__((slice(0, 3), 3), [1.0, -2.0, 3.5])),
    "pure_rotation_x": lambda: rotation_x(0.7),
    "pure_rotation_z": lambda: rotation_z(-2.1),
    "rotation_and_translation": lambda: _with(
        rotation_z(1.0), lambda m: m.__setitem__((slice(0, 3), 3), [0.1, 0.2, 0.3])
    ),
    "drifted_50_rotations": drifted_rotation,
}


def valid_cases():
    return [pytest.param(factory, id=name) for name, factory in VALID.items()]


def make_world():
    g1 = pp.CollisionGroup("robot", [pp.Box([0.5, 0.5, 0.5])])
    static_tf = np.eye(4, dtype=np.float64)
    g2 = pp.CollisionGroup("obstacle", [pp.Box([0.5, 0.5, 0.5])], static=True, transform=static_tf)
    return pp.CollisionWorld([g1, g2])


class TestRejection:
    """Every entry point rejects a non-rigid transform with a ValueError."""

    @pytest.mark.parametrize(("factory", "fragment"), invalid_cases())
    def test_collision_object(self, factory, fragment):
        with pytest.raises(ValueError, match=fragment):
            pp.CollisionObject(pp.Box([0.5, 0.5, 0.5]), transform=factory())

    @pytest.mark.parametrize(("factory", "fragment"), invalid_cases())
    def test_collision_group_static_transform(self, factory, fragment):
        with pytest.raises(ValueError, match=fragment):
            pp.CollisionGroup("g", [pp.Box([0.5, 0.5, 0.5])], static=True, transform=factory())

    @pytest.mark.parametrize(("factory", "fragment"), invalid_cases())
    def test_check_single(self, factory, fragment):
        world = make_world()
        with pytest.raises(ValueError, match=fragment):
            world.check({"robot": factory()}, [("robot", "obstacle", 0.0)])

    @pytest.mark.parametrize(("factory", "fragment"), invalid_cases())
    def test_check_batch(self, factory, fragment):
        """The (N, 4, 4) batch path builds transforms itself - it must check too."""
        world = make_world()
        batch = np.tile(np.eye(4), (4, 1, 1)).astype(np.float64)
        batch[2] = factory()
        with pytest.raises(ValueError, match=fragment):
            world.check({"robot": batch}, [("robot", "obstacle", 0.0)])

    def test_batch_error_names_the_index(self):
        world = make_world()
        batch = np.tile(np.eye(4), (4, 1, 1)).astype(np.float64)
        batch[2, 0, 3] = np.nan
        with pytest.raises(ValueError, match="index 2"):
            world.check({"robot": batch}, [("robot", "obstacle", 0.0)])

    def test_error_message_shows_the_offending_value(self):
        with pytest.raises(ValueError) as excinfo:
            pp.CollisionObject(pp.Box([0.5, 0.5, 0.5]), transform=np.diag([2.0, 2.0, 2.0, 1.0]))
        message = str(excinfo.value)
        # A 2x scale makes (R^T R)[0][0] = 4; the message must show that value
        # rather than just saying "invalid transform".
        assert "orthonormal" in message
        assert "4" in message


class TestAcceptance:
    """Legitimate rigid transforms keep working - including drifted ones."""

    @pytest.mark.parametrize("factory", valid_cases())
    def test_collision_object(self, factory):
        pp.CollisionObject(pp.Box([0.5, 0.5, 0.5]), transform=factory())

    @pytest.mark.parametrize("factory", valid_cases())
    def test_collision_group_static_transform(self, factory):
        pp.CollisionGroup("g", [pp.Box([0.5, 0.5, 0.5])], static=True, transform=factory())

    @pytest.mark.parametrize("factory", valid_cases())
    def test_check_single(self, factory):
        world = make_world()
        result = world.check({"robot": factory()}, [("robot", "obstacle", 0.0)])
        assert result.shape == (1,)

    @pytest.mark.parametrize("factory", valid_cases())
    def test_check_batch(self, factory):
        world = make_world()
        batch = np.tile(factory(), (3, 1, 1)).astype(np.float64)
        result = world.check({"robot": batch}, [("robot", "obstacle", 0.0)])
        assert result.shape[0] == 3

    def test_drift_is_well_inside_the_tolerance(self):
        """The 1e-6 tolerance must sit far above realistic accumulated drift.

        If this starts failing, the tolerance is not the thing to loosen - the
        drift model or the construction is wrong.
        """
        r = drifted_rotation(50)[:3, :3]
        orthonormality_error = np.abs(r.T @ r - np.eye(3)).max()
        determinant_error = abs(np.linalg.det(r) - 1.0)
        assert orthonormality_error < 1e-9, orthonormality_error
        assert determinant_error < 1e-9, determinant_error

    def test_rotated_pose_still_answers_correctly(self):
        """Validation must not change the answer for a valid transform."""
        world = make_world()
        far = np.eye(4, dtype=np.float64)
        far[0, 3] = 5.0
        assert not world.check({"robot": far}, [("robot", "obstacle", 0.0)])[0]

        near = rotation_z(0.3)
        near[0, 3] = 0.2
        assert world.check({"robot": near}, [("robot", "obstacle", 0.0)])[0]
