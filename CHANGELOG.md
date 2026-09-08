# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.4] - unreleased

### Fixed

- **`check` / `check_any` now validate the shape of every `pairs` entry**
  ([#17]). Each entry must be a `(group_a, group_b, min_distance)` 3-tuple;
  anything else raises `ValueError` naming the offending index, e.g.
  `pairs[3]: expected a (group_a, group_b, min_distance) 3-tuple, got 4
  elements`. Previously a too-short tuple raised a bare
  `IndexError: tuple index out of range` from inside the extension, naming
  neither the argument nor the pair.

  **Behaviour change:** an over-long tuple (4 or more elements) used to be
  accepted silently, with the extra elements discarded - the caller got a
  plausible-looking answer that ignored part of what was passed. It now
  raises. This fixes a bug rather than removing a deliberate API, but callers
  who were passing over-long tuples will see a new error.

- Pair element types are validated too: a non-`str` group name or a
  non-numeric `min_distance` raises `TypeError` naming the index and the
  field (`pairs[0]: min_distance must be a float, got 'str'`) instead of the
  contextless `must be real number, not str`.

## [0.0.3] - 2026-09-08

### Changed

- **BREAKING: non-rigid and non-finite 4x4 transforms now raise `ValueError`**
  ([#12], [#16]). Transforms that were previously accepted and silently
  produced a wrong answer are now rejected at the Python boundary. Every entry
  point that takes a transform validates it - `CollisionObject(transform=...)`,
  `CollisionGroup(transform=...)`, and both the `(4, 4)` and `(N, 4, 4)` paths
  of `check` / `check_any`. Three checks, in order:

  1. every one of the 16 entries is finite - no `NaN`, no `Inf`;
  2. the bottom row is `[0, 0, 0, 1]` (absolute tolerance 1e-12);
  3. the upper-left 3x3 block is a proper rotation - `R^T R = I` and
     `det(R) = +1` within 1e-6. A determinant of -1 is a reflection and is
     rejected, since it would mirror the geometry.

  The error message names the check that failed and shows the offending value;
  batch errors name the group and the pose index.

  Callers passing a genuine rotation are unaffected. The 1e-6 orthonormality
  tolerance sits nine orders of magnitude above the drift of a rotation
  accumulated through a 50-link forward-kinematics chain (measured at ~1e-15
  for f64) and far below a real error such as a 2x scale, which is off by 1.0.
  A caller that was relying on scale or shear being silently dropped, or on a
  `NaN` pose being answered, now gets an exception.

### Fixed

- **`matrix4_to_isometry` silently accepted non-rigid transforms** ([#12]).
  It assumes a rigid transform but nothing validated one, and
  `extract_transform_4x4` checked only the `(4, 4)` shape. Scale and shear were
  discarded by `Rot3::from_mat3` (documented as ill-defined for such a block,
  and only panicking with glam's `glam_assert` feature, which is off here), the
  bottom row was never read, and `NaN`/`Inf` propagated straight into the pose.
  For a collision checker the failure direction is the dangerous one: a matrix
  with a 2x scale and a 1.2 translation returned "no collision" where a real 2x
  box would have collided, and a `NaN` pose returned "collision" from garbage.
  The realistic trigger is not a deliberate scale matrix but a `NaN` from a
  diverged IK solve, an uninitialised pose, or a bad interpolation. This
  predates the glam migration - the nalgebra implementation used
  `Rotation3::from_matrix_unchecked`.

  Validation lives at the Python boundary, not in `matrix4_to_isometry`, which
  runs once per pose per group inside the Rayon query loop; a caller-supplied
  transform crosses the boundary exactly once, so the check is off the hot
  path. Transforms restored by `CollisionWorld.from_bytes` are not re-checked -
  they were validated when the world was built.

### Documentation

- The rigid-transform contract is now stated in `README.md`, `EXAMPLES.md` and
  `DESIGN.md`, and the README quick start no longer uses
  `np.random.rand(N, 4, 4)` as a placeholder - it is not a rigid transform and
  would now raise.

[#16]: https://github.com/CEAD-group/py-parry3d/pull/16

## [0.0.2] - 2026-09-08

### Fixed

- **A `CollisionGroup` holding a single object ignored that object's local
  transform** ([#1], [#2]). `build_shape` returned the bare shape for a lone
  object and the query applied only the group pose to it, so the object's own
  offset was silently dropped and the shape was tested centred on the group
  frame. Multi-object groups were unaffected, because their `Compound` carries
  the per-object isometries. `build_shape` now returns the lone object's
  isometry alongside its shape, and both query sites compose it onto the group
  pose.

- **A `Cylinder` in a multi-object `CollisionGroup` panicked** ([#9], [#14])
  with `Nested composite shapes are not allowed`, raised as
  `pyo3_runtime.PanicException` at `CollisionWorld` construction. The Y-to-Z
  reorientation of parry's cylinder was expressed as a one-element `Compound`,
  which nested inside the group's own `Compound`. Shapes now carry an intrinsic
  pose offset that is composed into the object's isometry instead, so cylinders
  no longer go through composite dispatch at all.

- **A false-negative collision band for cylinders**, fixed upstream in parry
  0.30.1 and picked up by the dependency upgrade below. A lone cylinder group
  tested against a multi-shape group reported "clear" for separations in
  roughly 0.505-0.575 while both smaller and larger separations up to the true
  contact distance reported colliding - non-monotone in separation, so a caller
  sweeping poses could see an isolated "safe" verdict surrounded by unsafe ones.

### Changed

- Upgraded `parry3d-f64` from 0.25.3 to 0.30.2 ([#5], [#6]). Parry dropped
  nalgebra in favour of glam as of 0.26.0, so this replaces `Isometry3`,
  `Point3`/`Vector3`, `UnitQuaternion` and `Rotation3` with parry's `Pose3`,
  `Vec3`, `Rot3` and `Mat3`, and removes the `nalgebra` dependency entirely.
  The Python-facing API is unchanged: transforms are still 4x4 row-major NumPy
  arrays.
- Upgraded `pyo3` 0.27 to 0.29, `numpy` 0.27 to 0.29 and `rayon` 1.10 to 1.12
  ([#3]).

### Compatibility

- **Serialized `CollisionWorld` blobs written by 0.0.1 still load.** Only
  plain data is persisted (`cached_shape` is `#[serde(skip)]` and rebuilt on
  deserialize), so neither the glam migration nor the cylinder fix changes the
  format. Verified by writing blobs on the old build and loading them on the
  new one.
- Note that `to_bytes()` output has never been byte-reproducible across runs -
  group names are emitted in `HashMap` order - so blobs should be compared by
  behaviour, not by hash. This is unchanged, not new.

### Documentation

- Documented `matrix4_to_isometry`'s rigid-transform contract ([#13]). It
  assumes a finite, row-major, rigid 4x4 with no scale or shear; violations are
  silently accepted and produce a wrong rotation rather than an error. See
  [#12] for the open question of whether to validate.

### Internal

- CI actually runs its test matrix again ([#4]). The lint job installed `ruff`
  unpinned, a newer release widened its default rule set, and because lint gates
  every other job the entire build and test matrix had been silently skipped.
  `ruff` is now pinned.
- The `test` job selects the exact ABI wheel ([#4]); `*cp314*` also matched the
  free-threaded `cp314t` build.
- The `release` job only creates a GitHub Release on tag refs ([#7], [#10]),
  making a `workflow_dispatch` dry run of the wheel matrix safe.
- `linux-x64` builds its wheels once rather than three times ([#8], [#11]).

[#1]: https://github.com/CEAD-group/py-parry3d/issues/1
[#2]: https://github.com/CEAD-group/py-parry3d/pull/2
[#3]: https://github.com/CEAD-group/py-parry3d/pull/3
[#4]: https://github.com/CEAD-group/py-parry3d/pull/4
[#5]: https://github.com/CEAD-group/py-parry3d/pull/5
[#6]: https://github.com/CEAD-group/py-parry3d/pull/6
[#7]: https://github.com/CEAD-group/py-parry3d/issues/7
[#8]: https://github.com/CEAD-group/py-parry3d/issues/8
[#9]: https://github.com/CEAD-group/py-parry3d/issues/9
[#10]: https://github.com/CEAD-group/py-parry3d/pull/10
[#11]: https://github.com/CEAD-group/py-parry3d/pull/11
[#12]: https://github.com/CEAD-group/py-parry3d/issues/12
[#13]: https://github.com/CEAD-group/py-parry3d/pull/13
[#14]: https://github.com/CEAD-group/py-parry3d/pull/14
[#17]: https://github.com/CEAD-group/py-parry3d/issues/17

## [0.0.1] - 2026-01-08

Initial release.

[0.0.3]: https://github.com/CEAD-group/py-parry3d/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/CEAD-group/py-parry3d/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/CEAD-group/py-parry3d/releases/tag/v0.0.1
