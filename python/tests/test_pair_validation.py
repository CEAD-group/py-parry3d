"""Tests for `pairs` argument validation in check/check_any (issue #17)."""

import numpy as np
import py_parry3d as pp
import pytest


def _world() -> pp.CollisionWorld:
    return pp.CollisionWorld(
        [
            pp.CollisionGroup("a", [pp.Sphere(0.1)]),
            pp.CollisionGroup("b", [pp.Sphere(0.1)]),
        ]
    )


def _transforms() -> dict[str, np.ndarray]:
    return {"a": np.eye(4), "b": np.eye(4)}


class TestPairArity:
    """Pair tuples must have exactly three elements."""

    @pytest.mark.parametrize("method", ["check", "check_any"])
    @pytest.mark.parametrize(
        ("pair", "n_elements"),
        [
            (("a", "b"), 2),
            (("a", "b", 0.0, 1.0), 4),
            (("a",), 1),
            ((), 0),
        ],
    )
    def test_wrong_arity_raises(
        self, method: str, pair: tuple, n_elements: int
    ) -> None:
        world = _world()
        with pytest.raises(ValueError) as exc:
            getattr(world, method)(_transforms(), [("a", "b", 0.0), pair])
        message = str(exc.value)
        assert "pairs[1]" in message
        assert f"got {n_elements} elements" in message

    @pytest.mark.parametrize("method", ["check", "check_any"])
    @pytest.mark.parametrize("pair", ["ab", ["a", "b", 0.0], None, 3])
    def test_non_tuple_raises(self, method: str, pair: object) -> None:
        world = _world()
        with pytest.raises(ValueError) as exc:
            getattr(world, method)(_transforms(), [pair])
        assert "pairs[0]" in str(exc.value)


class TestPairElementTypes:
    """Pair elements must be (str, str, float)."""

    @pytest.mark.parametrize("method", ["check", "check_any"])
    @pytest.mark.parametrize(
        ("pair", "field"),
        [
            ((1, "b", 0.0), "group_a"),
            (("a", None, 0.0), "group_b"),
            (("a", "b", "close"), "min_distance"),
            (("a", "b", None), "min_distance"),
        ],
    )
    def test_bad_element_type_raises(
        self, method: str, pair: tuple, field: str
    ) -> None:
        world = _world()
        with pytest.raises(TypeError) as exc:
            getattr(world, method)(_transforms(), [pair])
        message = str(exc.value)
        assert "pairs[0]" in message
        assert field in message


class TestValidPairs:
    """The valid paths are unchanged."""

    def test_valid_triple_still_works(self) -> None:
        world = _world()
        result = np.asarray(world.check(_transforms(), [("a", "b", 0.0)]))
        assert result.shape == (1,)
        assert bool(result[0])
        assert world.check_any(_transforms(), [("a", "b", 0.0)]) == 0

    def test_empty_pairs_list(self) -> None:
        world = _world()
        result = np.asarray(world.check(_transforms(), []))
        assert result.shape == (0,)
        assert world.check_any(_transforms(), []) is None
