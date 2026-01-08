# py-parry3d Examples

> Architecture overview: [DESIGN.md](DESIGN.md)

Implementation examples and API usage.

## Shapes

```python
import py_parry3d as pp
import numpy as np

# Primitives (centered at origin)
box = pp.Box(half_extents=[0.5, 0.3, 0.2])
sphere = pp.Sphere(radius=0.1)
capsule = pp.Capsule(half_height=0.5, radius=0.1)  # along Z axis
cylinder = pp.Cylinder(half_height=0.5, radius=0.1)  # along Z axis

# Mesh from vertices and faces
vertices = np.array([...], dtype=np.float64)  # (N, 3)
faces = np.array([...], dtype=np.uint32)      # (M, 3)
mesh = pp.TriMesh(vertices, faces)

# From trimesh object (convenient for loading mesh files)
import trimesh
tm = trimesh.load("part.stl")
mesh = pp.TriMesh.from_trimesh(tm)  # Extracts vertices and faces

# Convex hull (faster collision detection than TriMesh)
hull = pp.ConvexHull_from_trimesh(tm)  # Computes convex hull in Rust
# Access hull geometry for visualization
hull.vertices  # (N, 3) float64 - hull vertices
hull.faces     # (M, 3) uint32 - triangulated hull faces
```

## Transforms

```python
from scipy.spatial.transform import RigidTransform, Rotation

# From translation only
tf = RigidTransform.from_translation([0.5, 0, -0.05])

# From rotation only
tf = RigidTransform.from_rotation(Rotation.from_euler('z', 45, degrees=True))

# From both
tf = RigidTransform.from_components(
    rotation=Rotation.from_euler('z', 45, degrees=True),
    translation=[0.5, 0, -0.05],
)

# Identity
tf = RigidTransform.identity()

# Compose transforms
tf = tf1 * tf2

# py-parry3d accepts RigidTransform directly (calls .as_matrix() internally)
# Raw 4x4 numpy arrays also accepted for compatibility
```

## CollisionObject

```python
from scipy.spatial.transform import RigidTransform

# Shape at group origin (no transform needed)
obj1 = pp.CollisionObject(pp.Capsule(0.4, 0.08))

# Shape with local offset
obj2 = pp.CollisionObject(
    pp.Sphere(0.05),
    transform=RigidTransform.from_translation([0, 0, 0.3])
)

# Shorthand: bare shapes are wrapped automatically with identity transform
# pp.Capsule(0.4, 0.08) is equivalent to pp.CollisionObject(pp.Capsule(0.4, 0.08))
```

## CollisionGroup

```python
# Dynamic group - transform provided at check time
robot_l1 = pp.CollisionGroup("robot_l1", [
    pp.Capsule(0.4, 0.08),  # bare shape at group origin
    pp.CollisionObject(pp.Sphere(0.05), transform=tip_offset),  # with offset
])

# Static group - transform fixed at creation
table = pp.CollisionObject(pp.Box([1.0, 2.0, 0.05]))
obstacle = pp.CollisionObject(pp.Box([0.3, 0.3, 0.4]), transform=obstacle_local_tf)

environment = pp.CollisionGroup("environment", [table, obstacle],
    static=True,
    transform=RigidTransform.from_translation([0.5, 0, 0])
)
```

## CollisionWorld

```python
world = pp.CollisionWorld([
    robot_l1, robot_l2, robot_l3, robot_l4, robot_l5, robot_l6, robot_l7,
    environment,
])

# Properties
world.dynamic_groups   # list of dynamic group names
world.static_groups    # list of static group names
```

## Collision Pairs

```python
# Pairs with minimum distance threshold (safety margin)
# Objects closer than min_distance are considered "colliding"
pairs = [
    ("robot_l1", "robot_l3", 0.0),      # actual intersection only
    ("robot_l1", "environment", 0.05),  # 5cm safety margin
    ("tool", "environment", 0.10),      # 10cm margin for tool
]

# Helpers for common patterns
robot_links = ["robot_l1", "robot_l2", "robot_l3", "robot_l4", "robot_l5", "robot_l6", "robot_l7"]

# All pairs between groups, skipping adjacent (for robot self-collision)
pairs = pp.all_pairs(robot_links, skip_adjacent=1)  # min_distance=0

# With uniform distance threshold
pairs = pp.all_pairs(robot_links, skip_adjacent=1, min_distance=0.02)

# All dynamic groups vs a static group
pairs += pp.pairs_vs(robot_links, "environment")  # min_distance=0
pairs += pp.pairs_vs(robot_links, "environment", min_distance=0.05)
```

## Collision Check

```python
# Transforms: dict mapping group name → (N, 4, 4) array
# Static groups are omitted (they use their fixed transform)
N = 1_000_000

transforms = {
    "robot_l1": l1_transforms,  # (N, 4, 4) array
    "robot_l2": l2_transforms,
    "robot_l3": l3_transforms,
    # ... all dynamic groups
}

# Batch collision check
collisions = world.check(transforms, pairs)
# Returns: (N, n_pairs) bool array - True if pair collides for that pose

# Check if any collision per pose
any_collision = collisions.any(axis=1)  # (N,) bool array

# Count poses with collisions
n_colliding_poses = any_collision.sum()

# Single pose: same API, just (4, 4) arrays instead of (N, 4, 4)
single_transforms = {
    "robot_l1": l1_tf,  # (4, 4) array
    "robot_l2": l2_tf,
    # ...
}
collisions = world.check(single_transforms, pairs)  # (n_pairs,) bool array
```

## Early-Exit Check

```python
# Stop at first collision found (faster for validation)
result = world.check_any(transforms, pairs)
# Returns: Optional[int] - index of first pose with collision, or None

if result is not None:
    print(f"Collision found at pose {result}")
else:
    print("Path is collision-free")

# Note: Due to parallel execution, this may not return the
# chronologically-first collision. It returns the first collision
# found by any thread. For paths where the goal is no collisions,
# this is sufficient and much faster than full batch check.
```

## Memory Layout

```python
# All arrays must be C-contiguous, float64 for transforms/vertices, uint32 for indices
transforms = np.ascontiguousarray(transforms, dtype=np.float64)
pairs = np.ascontiguousarray(pairs, dtype=np.uint32)
```

## Threading

```python
# Set number of parallel threads (default: all cores)
pp.set_num_threads(8)

# Get current thread count
n = pp.get_num_threads()
```

## Serialization

```python
# Serialize to bytes
data: bytes = world.to_bytes()

# Save to file (caller handles caching)
with open("collision_world.bin", "wb") as f:
    f.write(data)

# Load from bytes
with open("collision_world.bin", "rb") as f:
    world = pp.CollisionWorld.from_bytes(f.read())

# Pickle protocol also supported
import pickle
pickle.dump(world, open("world.pkl", "wb"))
world = pickle.load(open("world.pkl", "rb"))
```

## Error Handling

```python
# Missing transform for dynamic group → ValueError
world.check({"robot_l1": ...}, pairs)  # missing other groups

# Unknown group name in pairs → ValueError
pairs = [("robot_l1", "nonexistent", 0.0)]

# No NaNs allowed - caller must filter invalid poses before collision checking
```

## Full Example: Robot vs Environment

```python
import numpy as np
import py_parry3d as pp
from scipy.spatial.transform import RigidTransform

# Define robot link collision shapes
link_shapes = [
    (0.2, 0.1), (0.4, 0.08), (0.5, 0.07),
    (0.3, 0.06), (0.6, 0.05), (0.1, 0.06), (0.05, 0.04)
]
robot_groups = [
    pp.CollisionGroup(f"robot_l{i}", [pp.Capsule(half_h, radius)])
    for i, (half_h, radius) in enumerate(link_shapes)
]

# Define environment (static)
environment = pp.CollisionGroup("environment", [
    pp.CollisionObject(pp.Box([1.0, 2.0, 0.05]),
        transform=RigidTransform.from_translation([0.5, 0, -0.05])),
    pp.CollisionObject(pp.Box([0.3, 0.3, 0.4]),
        transform=RigidTransform.from_translation([0.3, 0.5, 0.2])),
], static=True)

# Create world
world = pp.CollisionWorld(robot_groups + [environment])

# Define collision pairs
link_names = [f"robot_l{i}" for i in range(7)]
pairs = pp.all_pairs(link_names, skip_adjacent=1)              # self-collision (no margin)
pairs += pp.pairs_vs(link_names, "environment", min_distance=0.03)  # 3cm safety margin

# Get transforms from py-opw-kinematics
# batch_forward_frames() returns (N, 7, 4, 4)
fk_transforms = ...  # from py-opw-kinematics

# Convert to dict (one entry per dynamic group)
transforms = {
    f"robot_l{i}": fk_transforms[:, i, :, :]
    for i in range(7)
}

# Check collisions
collisions = world.check(transforms, pairs)
# Returns: (N, n_pairs) bool array
any_collision = collisions.any(axis=1)  # (N,) bool
print(f"Collisions: {any_collision.sum()} / {len(fk_transforms)}")
```
