"""Type stubs for py_parry3d._internal Rust extension."""

from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import numpy.typing as npt

# Type aliases
Transform = npt.NDArray[np.float64]  # (4, 4) array
BatchTransform = npt.NDArray[np.float64]  # (N, 4, 4) array
TransformDict = Dict[str, Union[Transform, BatchTransform]]
PairList = List[tuple[str, str, float]]

# ============================================================================
# Shapes
# ============================================================================

class Box:
    """A box shape with given half-extents."""

    def __init__(self, half_extents: Sequence[float]) -> None:
        """
        Create a box shape.

        :param half_extents: Half-extents [x, y, z] in meters.
        """
        ...

    def __repr__(self) -> str: ...


class Sphere:
    """A sphere shape with given radius."""

    def __init__(self, radius: float) -> None:
        """
        Create a sphere shape.

        :param radius: Radius in meters.
        """
        ...

    def __repr__(self) -> str: ...


class Capsule:
    """A capsule shape (cylinder with hemispherical caps) along Z axis."""

    def __init__(self, half_height: float, radius: float) -> None:
        """
        Create a capsule shape.

        :param half_height: Half-height of the cylindrical part in meters.
        :param radius: Radius in meters.
        """
        ...

    def __repr__(self) -> str: ...


class Cylinder:
    """A cylinder shape along Z axis."""

    def __init__(self, half_height: float, radius: float) -> None:
        """
        Create a cylinder shape.

        :param half_height: Half-height in meters.
        :param radius: Radius in meters.
        """
        ...

    def __repr__(self) -> str: ...


class TriMesh:
    """A triangle mesh shape."""

    def __init__(
        self,
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.uint32],
    ) -> None:
        """
        Create a triangle mesh shape.

        :param vertices: (N, 3) array of vertex positions.
        :param faces: (M, 3) array of triangle indices.
        """
        ...

    def __repr__(self) -> str: ...


class ConvexHull:
    """A convex hull collision shape computed from mesh vertices."""

    @staticmethod
    def from_mesh(
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.uint32],
    ) -> "ConvexHull":
        """
        Create a convex hull from mesh vertices and faces.

        :param vertices: (N, 3) array of vertex positions.
        :param faces: (M, 3) array of triangle indices.
        :return: A ConvexHull shape.
        """
        ...

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        """Convex hull vertices (N, 3) float64 array."""
        ...

    @property
    def faces(self) -> npt.NDArray[np.uint32]:
        """Convex hull faces (M, 3) uint32 array."""
        ...

    def __repr__(self) -> str: ...


# Type alias for any shape
Shape = Union[Box, Sphere, Capsule, Cylinder, TriMesh, ConvexHull]


# ============================================================================
# Core Types
# ============================================================================

class CollisionObject:
    """A shape with an optional local transform."""

    def __init__(
        self,
        shape: Shape,
        transform: Optional[Transform] = None,
    ) -> None:
        """
        Create a collision object.

        :param shape: The collision shape.
        :param transform: Optional local transform (4x4 matrix). Default: identity.
        """
        ...

    def __repr__(self) -> str: ...


class CollisionGroup:
    """A named group of collision objects sharing the same transform."""

    name: str
    is_static: bool

    def __init__(
        self,
        name: str,
        objects: List[Union[Shape, CollisionObject]],
        is_static: bool = False,
        transform: Optional[Transform] = None,
    ) -> None:
        """
        Create a collision group.

        :param name: Unique name for this group.
        :param objects: List of shapes or CollisionObjects in this group.
        :param is_static: If True, this group has a fixed transform.
        :param transform: Required for static groups - the fixed world transform.
        """
        ...

    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...


class CollisionWorld:
    """Container for all collision groups."""

    @property
    def dynamic_groups(self) -> List[str]:
        """List of dynamic group names."""
        ...

    @property
    def static_groups(self) -> List[str]:
        """List of static group names."""
        ...

    def __init__(self, groups: List[CollisionGroup]) -> None:
        """
        Create a collision world.

        :param groups: List of CollisionGroups (dynamic and static).
        """
        ...

    def __len__(self) -> int: ...

    def check(
        self,
        transforms: TransformDict,
        pairs: PairList,
    ) -> npt.NDArray[np.bool_]:
        """
        Check collisions for given transforms and pairs.

        :param transforms: Dict mapping dynamic group name to transform array.
        :param pairs: List of (group_a, group_b, min_distance) tuples to check.
        :return: Boolean array. Single pose: (n_pairs,). Batch: (N, n_pairs).
        """
        ...

    def check_any(
        self,
        transforms: TransformDict,
        pairs: PairList,
    ) -> Optional[int]:
        """
        Check for any collision, returning early on first hit.

        :param transforms: Dict mapping dynamic group name to transform array.
        :param pairs: List of (group_a, group_b, min_distance) tuples to check.
        :return: Index of first pose with collision, or None if no collisions.
        """
        ...

    def to_bytes(self) -> bytes:
        """Serialize the world to bytes (includes pre-built BVHs)."""
        ...

    @staticmethod
    def from_bytes(data: bytes) -> "CollisionWorld":
        """Deserialize a world from bytes."""
        ...


# ============================================================================
# Helper Functions
# ============================================================================

def all_pairs(
    groups: List[str],
    skip_adjacent: int = 0,
    min_distance: float = 0.0,
) -> PairList:
    """
    Generate all pairs between groups.

    :param groups: List of group names.
    :param skip_adjacent: Skip pairs where |i - j| <= skip_adjacent.
    :param min_distance: Minimum distance threshold for all pairs.
    :return: List of (group_a, group_b, min_distance) tuples.
    """
    ...


def pairs_vs(
    groups: List[str],
    other: Union[str, List[str]],
    min_distance: float = 0.0,
) -> PairList:
    """
    Generate pairs between groups and other group(s).

    :param groups: List of group names.
    :param other: Single group name or list of group names.
    :param min_distance: Minimum distance threshold for all pairs.
    :return: List of (group_a, group_b, min_distance) tuples.
    """
    ...


def transform(
    rotation: Optional[object] = None,
    translation: Optional[Sequence[float]] = None,
) -> Transform:
    """
    Create a 4x4 transform matrix.

    :param rotation: Object with as_matrix() method (e.g., scipy.spatial.transform.Rotation).
    :param translation: Translation [x, y, z].
    :return: (4, 4) transform matrix.
    """
    ...


def set_num_threads(n: int) -> bool:
    """
    Set the number of threads for parallel operations.

    Must be called BEFORE the first parallel operation (check/check_any).
    Once the thread pool is initialized, this function has no effect.

    :param n: Number of threads to use.
    :return: True if successful, False if thread pool was already initialized.
    """
    ...


def get_num_threads() -> int:
    """Get the current number of threads for parallel operations."""
    ...
