"""
py-parry3d: Python bindings for parry3d collision detection.

Optimized for batch operations with NumPy arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, Union

import numpy as np
import numpy.typing as npt

from . import _internal

if TYPE_CHECKING:
    import trimesh  # type: ignore[import-not-found]


# Type aliases
Transform = npt.NDArray[np.float64]  # (4, 4) array
BatchTransform = npt.NDArray[np.float64]  # (N, 4, 4) array
TransformLike = Union[Transform, "RigidTransformProtocol"]
TransformDict = dict[str, Union[Transform, BatchTransform]]
PairList = list[tuple[str, str, float]]



class RigidTransformProtocol(Protocol):
    """Protocol for scipy.spatial.transform.RigidTransform."""

    def as_matrix(self) -> npt.NDArray[np.float64]: ...


def _to_matrix(transform: Optional[TransformLike]) -> Optional[npt.NDArray[np.float64]]:
    """Convert transform to 4x4 matrix, handling RigidTransform."""
    if transform is None:
        return None
    # Check for RigidTransform-like objects with as_matrix method
    as_matrix = getattr(transform, "as_matrix", None)
    if as_matrix is not None:
        return np.asarray(as_matrix(), dtype=np.float64)
    return np.asarray(transform, dtype=np.float64)


# Re-export shapes directly (no wrapping needed)
Box = _internal.Box
Sphere = _internal.Sphere
Capsule = _internal.Capsule
Cylinder = _internal.Cylinder
_InternalTriMesh = _internal.TriMesh
_InternalConvexHull = _internal.ConvexHull


def TriMesh(vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.uint32]) -> _InternalTriMesh:
    """
    Create a triangle mesh shape.

    .. note:: **TriMesh is HOLLOW (surface-only)**

        TriMesh only detects collisions with the mesh surface. Objects fully
        inside the mesh will NOT be detected as colliding. This is different
        from ConvexHull which is solid.

        Use TriMesh when:
        - You need exact mesh geometry for collision
        - Surface contact detection is sufficient
        - Objects being inside the mesh is acceptable

        Use ConvexHull when:
        - You need solid collision detection
        - Simplified geometry is acceptable
        - You need to detect objects inside the shape

    :param vertices: (N, 3) array of vertex positions.
    :param faces: (M, 3) array of triangle indices.
    :return: A TriMesh shape.
    """
    return _InternalTriMesh(vertices, faces)


def TriMesh_from_trimesh(mesh: "trimesh.Trimesh") -> _InternalTriMesh:
    """
    Create a TriMesh from a trimesh.Trimesh object.

    .. note:: **TriMesh is HOLLOW (surface-only)**

        See :func:`TriMesh` for details on the difference between TriMesh
        and ConvexHull collision behavior.

    :param mesh: A trimesh.Trimesh object.
    :return: A TriMesh shape.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    return _InternalTriMesh(vertices, faces)


# Attach as class method-like
TriMesh.from_trimesh = staticmethod(TriMesh_from_trimesh)  # type: ignore


def ConvexHull_from_trimesh(mesh: "trimesh.Trimesh") -> _InternalConvexHull:
    """
    Create a convex hull collision shape from a trimesh.Trimesh object.

    The convex hull is computed in Rust using parry3d.
    Use .vertices and .faces properties to access geometry for visualization.

    .. note:: **ConvexHull is SOLID**

        ConvexHull detects collisions with objects both touching the surface
        AND fully inside the hull. This is different from TriMesh which only
        detects surface contact.

        Use ConvexHull when:
        - You need solid collision detection (e.g., robot safety)
        - Simplified convex geometry is acceptable
        - You need to detect objects inside the shape

        Use TriMesh when:
        - You need exact (possibly concave) mesh geometry
        - Surface contact detection is sufficient

    :param mesh: A trimesh.Trimesh object.
    :return: A ConvexHull shape.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    return _InternalConvexHull.from_mesh(vertices, faces)


# Type alias for any shape
Shape = Union[Box, Sphere, Capsule, Cylinder, _InternalTriMesh, _InternalConvexHull]


class CollisionObject:
    """A shape with an optional local transform."""

    def __init__(
        self,
        shape: Shape,
        transform: Optional[TransformLike] = None,
    ) -> None:
        """
        Create a collision object.

        :param shape: The collision shape.
        :param transform: Optional local transform (4x4 matrix or RigidTransform).
        """
        self._internal = _internal.CollisionObject(shape, _to_matrix(transform))

    def __repr__(self) -> str:
        return repr(self._internal)


class CollisionGroup:
    """A named group of collision objects sharing the same transform."""

    def __init__(
        self,
        name: str,
        objects: list[Union[Shape, CollisionObject]],
        static: bool = False,
        transform: Optional[TransformLike] = None,
    ) -> None:
        """
        Create a collision group.

        :param name: Unique name for this group.
        :param objects: List of shapes or CollisionObjects in this group.
        :param static: If True, this group has a fixed transform.
        :param transform: Required for static groups - the fixed world transform.
        """
        # Unwrap CollisionObject wrappers to internal type
        internal_objects = [
            obj._internal if isinstance(obj, CollisionObject) else obj for obj in objects
        ]
        self._internal = _internal.CollisionGroup(
            name, internal_objects, is_static=static, transform=_to_matrix(transform)
        )

    @property
    def name(self) -> str:
        return self._internal.name

    @property
    def is_static(self) -> bool:
        return self._internal.is_static

    def __repr__(self) -> str:
        return repr(self._internal)

    def __len__(self) -> int:
        return len(self._internal)


class CollisionWorld:
    """Container for all collision groups."""

    def __init__(self, groups: list[CollisionGroup]) -> None:
        """
        Create a collision world.

        :param groups: List of CollisionGroups (dynamic and static).
        """
        internal_groups = [g._internal for g in groups]
        self._internal = _internal.CollisionWorld(internal_groups)

    @property
    def dynamic_groups(self) -> list[str]:
        """List of dynamic group names."""
        return self._internal.dynamic_groups

    @property
    def static_groups(self) -> list[str]:
        """List of static group names."""
        return self._internal.static_groups

    def __len__(self) -> int:
        return len(self._internal)

    def check(
        self,
        transforms: TransformDict,
        pairs: PairList,
    ) -> npt.NDArray[np.bool_]:
        """
        Check collisions for given transforms and pairs.

        :param transforms: Dict mapping dynamic group name to transform array.
                          Single pose: (4, 4) array per group.
                          Batch: (N, 4, 4) array per group.
        :param pairs: List of (group_a, group_b, min_distance) tuples.
        :return: Boolean array. Single pose: (n_pairs,). Batch: (N, n_pairs).
        """
        return self._internal.check(transforms, pairs)

    def check_any(
        self,
        transforms: TransformDict,
        pairs: PairList,
    ) -> Optional[int]:
        """
        Check for any collision, returning early on first hit.

        :param transforms: Dict mapping dynamic group name to transform array.
        :param pairs: List of (group_a, group_b, min_distance) tuples.
        :return: Index of first pose with collision, or None if no collisions.
        """
        return self._internal.check_any(transforms, pairs)

    def to_bytes(self) -> bytes:
        """Serialize the world to bytes (includes pre-built BVHs)."""
        return self._internal.to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "CollisionWorld":
        """Deserialize a world from bytes."""
        world = cls.__new__(cls)
        world._internal = _internal.CollisionWorld.from_bytes(data)
        return world

    def __getstate__(self) -> bytes:
        return self.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        self._internal = _internal.CollisionWorld.from_bytes(state)


# Re-export helper functions
all_pairs = _internal.all_pairs
pairs_vs = _internal.pairs_vs
transform = _internal.transform
set_num_threads = _internal.set_num_threads
get_num_threads = _internal.get_num_threads


__all__ = [
    # Shapes
    "Box",
    "Sphere",
    "Capsule",
    "Cylinder",
    "TriMesh",
    "ConvexHull_from_trimesh",
    # Core types
    "CollisionObject",
    "CollisionGroup",
    "CollisionWorld",
    # Helper functions
    "all_pairs",
    "pairs_vs",
    "transform",
    "set_num_threads",
    "get_num_threads",
]
