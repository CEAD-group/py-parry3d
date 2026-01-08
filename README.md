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
transforms = {"robot": np.random.rand(N, 4, 4)}  # Your actual transforms here
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

### Collision Checking

```python
# Batch check
result = world.check(transforms, pairs)  # (N, n_pairs) bool array

# Early-exit (stops at first collision)
idx = world.check_any(transforms, pairs)  # int or None
```

### Pair Helpers

```python
# All pairs between groups
pairs = pp.all_pairs(["l1", "l2", "l3"], skip_adjacent=1)

# Groups vs target
pairs = pp.pairs_vs(["l1", "l2"], "obstacle", min_distance=0.05)
```

### Threading

```python
pp.set_num_threads(8)  # Default: all cores
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
