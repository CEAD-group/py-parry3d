"""Benchmarks for py-parry3d collision detection."""

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

import py_parry3d as pp


@dataclass
class BenchmarkResult:
    name: str
    batch_size: int
    num_pairs: int
    total_time_ms: float
    checks_per_second: float


def benchmark(func: Callable, warmup: int = 3, iterations: int = 10) -> float:
    """Run benchmark and return average time in seconds."""
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times)


def create_robot_world(num_links: int = 7) -> pp.CollisionWorld:
    """Create a typical robot collision world."""
    groups = []

    # Robot links (dynamic)
    for i in range(num_links):
        groups.append(pp.CollisionGroup(f"link_{i}", [pp.Capsule(0.15, 0.05)]))

    # Environment objects (static)
    floor_tf = np.eye(4, dtype=np.float64)
    floor_tf[2, 3] = -0.5
    groups.append(
        pp.CollisionGroup(
            "floor", [pp.Box([2.0, 2.0, 0.05])], static=True, transform=floor_tf
        )
    )

    table_tf = np.eye(4, dtype=np.float64)
    table_tf[0, 3] = 0.6
    table_tf[2, 3] = 0.4
    groups.append(
        pp.CollisionGroup(
            "table", [pp.Box([0.4, 0.3, 0.02])], static=True, transform=table_tf
        )
    )

    obstacle_tf = np.eye(4, dtype=np.float64)
    obstacle_tf[0, 3] = 0.3
    obstacle_tf[1, 3] = 0.2
    obstacle_tf[2, 3] = 0.5
    groups.append(
        pp.CollisionGroup(
            "obstacle", [pp.Sphere(0.1)], static=True, transform=obstacle_tf
        )
    )

    return pp.CollisionWorld(groups)


def create_mesh_world() -> pp.CollisionWorld:
    """Create a world with mesh shapes."""
    # Simple cube mesh
    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    ) * 0.1

    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 6, 5],
            [4, 7, 6],  # top
            [0, 4, 5],
            [0, 5, 1],  # front
            [2, 6, 7],
            [2, 7, 3],  # back
            [0, 3, 7],
            [0, 7, 4],  # left
            [1, 5, 6],
            [1, 6, 2],  # right
        ],
        dtype=np.uint32,
    )

    groups = []
    for i in range(5):
        groups.append(pp.CollisionGroup(f"mesh_{i}", [pp.TriMesh(vertices, faces)]))

    floor_tf = np.eye(4, dtype=np.float64)
    floor_tf[2, 3] = -1.0
    groups.append(
        pp.CollisionGroup(
            "floor", [pp.Box([5.0, 5.0, 0.1])], static=True, transform=floor_tf
        )
    )

    return pp.CollisionWorld(groups)


def generate_random_transforms(n: int, num_groups: int) -> dict:
    """Generate random transforms for benchmark."""
    transforms = {}
    for i in range(num_groups):
        tfs = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
        # Random positions in a cube
        tfs[:, :3, 3] = np.random.uniform(-1, 1, (n, 3))
        transforms[f"link_{i}"] = tfs
    return transforms


def run_batch_size_benchmark(world: pp.CollisionWorld, pairs: list, max_batch: int = 100_000):
    """Benchmark different batch sizes."""
    results = []
    num_dynamic = len(world.dynamic_groups)

    batch_sizes = [1, 10, 100, 1_000, 10_000, 50_000, 100_000]
    batch_sizes = [b for b in batch_sizes if b <= max_batch]

    for batch_size in batch_sizes:
        transforms = generate_random_transforms(batch_size, num_dynamic)

        def run():
            return world.check(transforms, pairs)

        elapsed = benchmark(run)
        total_checks = batch_size * len(pairs)
        checks_per_sec = total_checks / elapsed

        results.append(
            BenchmarkResult(
                name="batch_check",
                batch_size=batch_size,
                num_pairs=len(pairs),
                total_time_ms=elapsed * 1000,
                checks_per_second=checks_per_sec,
            )
        )

    return results


def run_pair_count_benchmark(world: pp.CollisionWorld, batch_size: int = 10_000):
    """Benchmark different numbers of collision pairs."""
    results = []
    num_dynamic = len(world.dynamic_groups)
    dynamic_names = world.dynamic_groups
    static_names = world.static_groups

    # Different pair configurations
    pair_configs = [
        ("self_collision", pp.all_pairs(dynamic_names, skip_adjacent=1)),
        ("vs_floor", pp.pairs_vs(dynamic_names, "floor")),
        ("vs_all_static", pp.pairs_vs(dynamic_names, static_names)),
        ("all_pairs", pp.all_pairs(dynamic_names) + pp.pairs_vs(dynamic_names, static_names)),
    ]

    transforms = generate_random_transforms(batch_size, num_dynamic)

    for name, pairs in pair_configs:
        if not pairs:
            continue

        def run():
            return world.check(transforms, pairs)

        elapsed = benchmark(run)
        total_checks = batch_size * len(pairs)
        checks_per_sec = total_checks / elapsed

        results.append(
            BenchmarkResult(
                name=name,
                batch_size=batch_size,
                num_pairs=len(pairs),
                total_time_ms=elapsed * 1000,
                checks_per_second=checks_per_sec,
            )
        )

    return results


def print_results(results: list[BenchmarkResult], title: str):
    """Print benchmark results in a table."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")
    print(f"{'Name':<20} {'Batch':>10} {'Pairs':>8} {'Time (ms)':>12} {'Checks/sec':>15}")
    print(f"{'-' * 70}")

    for r in results:
        print(
            f"{r.name:<20} {r.batch_size:>10,} {r.num_pairs:>8} "
            f"{r.total_time_ms:>12.2f} {r.checks_per_second:>15,.0f}"
        )


def main():
    print("py-parry3d Collision Detection Benchmarks")
    print(f"NumPy version: {np.__version__}")
    print(f"Threads: {pp.get_num_threads()}")

    # Robot world benchmarks
    print("\nCreating robot world (7-DOF arm + environment)...")
    robot_world = create_robot_world(num_links=7)
    robot_pairs = pp.all_pairs(robot_world.dynamic_groups, skip_adjacent=1) + pp.pairs_vs(
        robot_world.dynamic_groups, robot_world.static_groups
    )
    print(f"  Dynamic groups: {len(robot_world.dynamic_groups)}")
    print(f"  Static groups: {len(robot_world.static_groups)}")
    print(f"  Collision pairs: {len(robot_pairs)}")

    results = run_batch_size_benchmark(robot_world, robot_pairs)
    print_results(results, "Robot World - Batch Size Scaling")

    results = run_pair_count_benchmark(robot_world)
    print_results(results, "Robot World - Pair Count Comparison")

    # Mesh world benchmarks
    print("\nCreating mesh world (5 mesh objects + floor)...")
    mesh_world = create_mesh_world()
    mesh_pairs = pp.all_pairs(mesh_world.dynamic_groups) + pp.pairs_vs(
        mesh_world.dynamic_groups, mesh_world.static_groups
    )
    print(f"  Dynamic groups: {len(mesh_world.dynamic_groups)}")
    print(f"  Static groups: {len(mesh_world.static_groups)}")
    print(f"  Collision pairs: {len(mesh_pairs)}")

    # Generate transforms for mesh groups
    def generate_mesh_transforms(n: int) -> dict:
        transforms = {}
        for name in mesh_world.dynamic_groups:
            tfs = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
            tfs[:, :3, 3] = np.random.uniform(-1, 1, (n, 3))
            transforms[name] = tfs
        return transforms

    mesh_results = []
    for batch_size in [1, 100, 1_000, 10_000, 50_000]:
        transforms = generate_mesh_transforms(batch_size)

        def run():
            return mesh_world.check(transforms, mesh_pairs)

        elapsed = benchmark(run)
        total_checks = batch_size * len(mesh_pairs)

        mesh_results.append(
            BenchmarkResult(
                name="mesh_check",
                batch_size=batch_size,
                num_pairs=len(mesh_pairs),
                total_time_ms=elapsed * 1000,
                checks_per_second=total_checks / elapsed,
            )
        )

    print_results(mesh_results, "Mesh World - Batch Size Scaling")

    # Thread scaling benchmark
    print("\n" + "=" * 70)
    print(" Thread Scaling (batch=50,000, robot world)")
    print("=" * 70)

    transforms = generate_random_transforms(50_000, len(robot_world.dynamic_groups))
    max_threads = pp.get_num_threads()

    # Note: set_num_threads only works before first parallel op, so this is informational
    def run():
        return robot_world.check(transforms, robot_pairs)

    elapsed = benchmark(run)
    total_checks = 50_000 * len(robot_pairs)
    print(f"  Threads: {max_threads}")
    print(f"  Time: {elapsed * 1000:.2f} ms")
    print(f"  Throughput: {total_checks / elapsed:,.0f} checks/sec")

    # Summary
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"  Peak throughput (primitives): {results[-1].checks_per_second:,.0f} checks/sec")
    print(f"  Peak throughput (meshes): {mesh_results[-1].checks_per_second:,.0f} checks/sec")


if __name__ == "__main__":
    main()
