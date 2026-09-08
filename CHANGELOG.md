# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [0.0.1] - 2026-01-08

Initial release.

[0.0.2]: https://github.com/CEAD-group/py-parry3d/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/CEAD-group/py-parry3d/releases/tag/v0.0.1
