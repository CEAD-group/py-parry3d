# py-parry3d

Python bindings for [parry3d](https://parry.rs/) collision detection, optimized for batch operations with NumPy.

## Features

- **Batch-first**: Process millions of collision queries with minimal Python overhead
- **NumPy native**: Zero-copy data exchange using contiguous arrays
- **Parallel**: Automatic multi-threading via Rayon
- **Serializable**: Cache collision worlds with pre-built BVHs

## Installation

```bash
pip install py-parry3d
```

## Quick Start

```python
import numpy as np
import py_parry3d as pp

# Define collision shapes
robot_link = pp.CollisionGroup("robot", [pp.Capsule(half_height=0.4, radius=0.08)])
obstacle = pp.CollisionGroup("obstacle", [pp.Box(half_extents=[0.3, 0.3, 0.3])],
                              static=True, transform=np.eye(4))

# Create world
world = pp.CollisionWorld([robot_link, obstacle])

# Check collisions for 1000 robot poses
N = 1000
# Transforms must be rigid: a proper rotation in the 3x3 block, a
# [0, 0, 0, 1] bottom row, and no NaN/Inf. Anything else raises ValueError.
transforms = {"robot": np.tile(np.eye(4), (N, 1, 1))}  # Your actual poses here
pairs = [("robot", "obstacle", 0.0)]  # (group_a, group_b, min_distance)

collisions = world.check(transforms, pairs)  # (N, 1) bool array
print(f"Collisions: {collisions.sum()} / {N}")
```

## Shapes

```python
# Primitives
pp.Box(half_extents=[0.5, 0.3, 0.2])
pp.Sphere(radius=0.1)
pp.Capsule(half_height=0.5, radius=0.1)  # Z-axis aligned
pp.Cylinder(half_height=0.5, radius=0.1)  # Z-axis aligned

# Meshes
pp.TriMesh(vertices, faces)  # (N,3) float64, (M,3) uint32
pp.TriMesh.from_trimesh(mesh)  # From trimesh object

# Convex hull (faster than TriMesh)
pp.ConvexHull_from_trimesh(mesh)
```

### Solid vs Hollow

| Shape | Type | Inside Detection |
|-------|------|------------------|
| Box, Sphere, Capsule, Cylinder, ConvexHull | **Solid** | Detects objects inside |
| TriMesh | **Hollow** | Surface contact only |

## API

### Groups and World

```python
# Dynamic group - transform provided at check time
robot = pp.CollisionGroup("robot", [shape1, shape2])

# Static group - fixed transform
env = pp.CollisionGroup("env", [shape], static=True, transform=tf)

# World
world = pp.CollisionWorld([robot, env])
```

### Transforms

All 4x4 transforms - a `CollisionObject` local offset, a static group's
transform, and every pose passed to `check`/`check_any`/`check_first`/`check_per_pose` - must be **rigid**:

- the upper-left 3x3 block is a proper rotation (orthonormal, determinant +1);
  no scale, no shear, no reflection;
- the bottom row is `[0, 0, 0, 1]`;
- every entry is finite - no `NaN`, no `Inf`.

Violations raise `ValueError` naming the failed check and the offending value.
Orthonormality is checked to a tolerance of 1e-6, far above the float drift of
an accumulated forward-kinematics product (~1e-15) and far below a genuine
error such as a 2x scale.

### Collision Checking

```python
# Batch check
result = world.check(transforms, pairs)  # (N, n_pairs) bool array

# Early-exit (stops at a collision; index may vary between runs)
idx = world.check_any(transforms, pairs)  # int or None

# Lowest colliding pose plus which pairs collide there (deterministic)
first = world.check_first(transforms, pairs)  # (int, (n_pairs,) bool array) or None

# First colliding pair per pose, -1 where none (each pose stops at its first hit)
first_pair = world.check_per_pose(transforms, pairs)  # (N,) int64 array
hit = first_pair >= 0  # (N,) bool, same as check(...).any(axis=1)
```

### Pair Helpers

```python
# All pairs between groups
pairs = pp.all_pairs(["l1", "l2", "l3"], skip_adjacent=1)

# Groups vs target
pairs = pp.pairs_vs(["l1", "l2"], "obstacle", min_distance=0.05)
```

### Threading

The four batched `CollisionWorld` queries (`check`, `check_any`, `check_first`,
`check_per_pose`) are the only methods that run in parallel, so they are the
only ones that take `threads`. Two controls set how many threads they use:

- `set_num_threads(n)` sets the size of the process-wide default pool (default:
  all cores). It only works before the first parallel call and returns `False`
  after that.
- `threads=` (keyword-only) caps a single call and overrides the default for
  that call. `None` (the default) uses the process-wide pool, `1` runs serially
  in the calling thread without any pool, and `n >= 2` runs on a pool of `n`
  threads. `threads < 1` raises `ValueError`. The cap covers both transform
  validation and the query loop, and results do not depend on it, apart from
  which colliding pose `check_any` reports.

Pools for `threads >= 2` are built on first use and cached per thread count
for the life of the process, so their idle threads stay alive (asleep) between
calls, and each distinct value keeps its own pool.

```python
# Process-wide default, only before the first query
pp.set_num_threads(8)

# On a shared server, keep this call to 2 cores
hits = world.check(transforms, pairs, threads=2)
first_pair = world.check_per_pose(transforms, pairs, threads=2)

# Serial in the calling thread, e.g. when you parallelise across processes yourself
hits = world.check(transforms, pairs, threads=1)
```

### Serialization

```python
data = world.to_bytes()
world = pp.CollisionWorld.from_bytes(data)
# Pickle also supported
```

## Documentation

- [DESIGN.md](DESIGN.md) - Architecture and concepts
- [EXAMPLES.md](EXAMPLES.md) - Detailed usage examples

## License

MIT
