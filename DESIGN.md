# py-parry3d

> Implementation examples and code samples: [EXAMPLES.md](EXAMPLES.md)

Python bindings for [parry3d](https://parry.rs/) collision detection, optimized for batch operations with NumPy.

## Goals

- **Batch-first**: Process millions of collision queries with minimal Python overhead
- **NumPy native**: Zero-copy data exchange using contiguous arrays
- **Domain-agnostic**: No knowledge of robots, workcells, etc. - just geometry and transforms
- **Simple API**: Easy setup, fast hot path

## Non-Goals

- Physics simulation (use Rapier)
- 2D collision detection (use parry2d)
- Real-time continuous collision detection
- Ray casting / point queries (maybe later)
- Domain-specific logic (robots, etc.)

---

## Core Concepts

### Groups

py-parry3d checks collisions between **groups** of shapes.

- **Group**: One or more shapes that move together (share a transform). Each shape has a local offset within the group.
- **Static group**: Fixed transform (e.g., environment obstacles).
- **Dynamic group**: Transform varies per pose.

### Batch Operations

For batch operations:
1. Define groups of shapes (primitives or meshes)
2. Provide an array of transforms (one per dynamic group per pose)
3. Specify which group pairs to check
4. Get back a boolean array (one per pair per pose)

---

## API Overview

### Shapes

Collision geometry primitives (all dimensions in meters):
- `Box(half_extents)` - axis-aligned box
- `Sphere(radius)`
- `Capsule(half_height, radius)` - Z-axis aligned
- `Cylinder(half_height, radius)` - Z-axis aligned
- `TriMesh(vertices, faces)` - arbitrary triangle mesh with BVH
- `ConvexHull` - convex hull computed from mesh vertices (faster than TriMesh)

TriMesh and ConvexHull can be created from `trimesh` objects via `TriMesh.from_trimesh()` and `ConvexHull_from_trimesh()`. ConvexHull exposes `.vertices` and `.faces` properties for visualization.

#### Solid vs Hollow Shapes

**Important behavioral difference:**

| Shape | Treatment | Inside Detection |
|-------|-----------|-----------------|
| Box, Sphere, Capsule, Cylinder, **ConvexHull** | **SOLID** | Detects objects inside |
| **TriMesh** | **HOLLOW** | Only detects surface contact |

- **ConvexHull** (and all primitives) detect collisions with objects both touching the surface AND fully inside the shape
- **TriMesh** only detects collisions with the mesh surface; objects fully inside are NOT detected

**When to use which:**

| Use Case | Recommended Shape |
|----------|-------------------|
| Robot safety (must detect all contact) | ConvexHull or primitives |
| Exact concave geometry needed | TriMesh |
| Objects can legitimately be "inside" | TriMesh |
| Simplified geometry acceptable | ConvexHull (also faster) |

### Transforms

Uses `scipy.spatial.transform.RigidTransform` (scipy ≥1.16). Raw 4x4 numpy arrays also accepted.

Convention: 4x4 homogeneous matrices, column vectors (`transform @ point`).

### CollisionObject

A shape with an optional local transform (position within its group). Bare shapes are automatically wrapped with identity transform.

### CollisionGroup

Named collection of objects sharing the same transform. Can be static (fixed transform) or dynamic (transform provided at check time).

### CollisionWorld

Container for all groups. Immutable after creation.

### Collision Pairs

Specifies which group pairs to check, with optional `min_distance` threshold (safety margin). Objects closer than `min_distance` are considered "colliding".

Helper functions:
- `all_pairs(groups, skip_adjacent=0)` - all pairs, optionally skipping adjacent indices
- `pairs_vs(groups, target)` - all groups vs a single target group

### Collision Check

- `world.check(transforms, pairs)` → `(N, n_pairs)` bool array
- `world.check_any(transforms, pairs)` → first collision index or None (early-exit)

**NaN handling**: No NaNs allowed in transforms. Caller must filter invalid poses before collision checking.

---

## Memory Layout

All arrays must be:
- **C-contiguous** (row-major)
- **float64** for transforms/vertices
- **uint32** for indices

---

## Threading

Configurable thread count via `set_num_threads()`. Default: all cores.

---

## Serialization

CollisionWorld can be serialized to bytes (including pre-built BVHs) for caching. Caching logic (file storage, invalidation) is the caller's responsibility.

Supports both `to_bytes()`/`from_bytes()` and pickle protocol.

---

## Implementation Notes

### Rust Dependencies

- pyo3 (Python bindings)
- numpy (array interface)
- parry3d (collision detection)
- rayon (parallelism)
- nalgebra (linear algebra)
- serde + bincode (serialization)

### Key Optimizations

1. **BVH per mesh**: Each TriMesh gets its own bounding volume hierarchy
2. **Compound shapes for groups**: Groups with multiple shapes use parry3d's `Compound` shape with automatic BVH over sub-shapes
3. **Parallel iteration**: Rayon parallel iterator over poses
4. **No Python in hot loop**: All batch work happens in Rust
5. **Static group caching**: Static group transforms applied once, not per pose
6. **Distance thresholds**: `min_distance > 0` uses `distance()` query; `min_distance = 0` uses faster `intersection_test()`
7. **Early-exit**: `check_any()` uses `find_any()` to stop at first collision
