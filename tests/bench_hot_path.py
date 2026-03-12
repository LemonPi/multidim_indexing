"""Benchmarks for the optimization targets in OPTIMIZE.md.

Self-contained (no pytorch_seed dependency). Profiles both the overall
query path and individual methods to measure the impact of each optimization.

Run: python tests/bench_hot_path.py
"""

import time
import torch
import numpy as np
from multidim_indexing import torch_view as view
from multidim_indexing import numpy_view as np_view


def bench(fn, warmup=3, runs=20):
    """Run fn, return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times)
    return np.median(times), np.std(times)


def bench_nearest_value_range():
    """End-to-end nearest-neighbor with value ranges — the CachedSDF hot path."""
    print("=" * 60)
    print("Nearest-neighbor with value_ranges (CachedSDF hot path)")
    print("=" * 60)

    data = torch.randn(100, 100, 100)
    torch.manual_seed(0)

    for N in [20_000, 100_000]:
        query = torch.rand(N, 3) * torch.tensor([1.0, 5.0, 10.0])

        for safety in [True, False]:
            dv = view.TorchMultidimView(
                data, value_ranges=[(0, 1), (0, 5), (0, 10)],
                check_safety=safety,
            )
            med, std = bench(lambda: dv[query])
            label = "safe" if safety else "unsafe"
            print(f"  N={N:>7d}  {label:<6s}  {med:6.2f} ms (std {std:.2f})")
    print()


def bench_linear_value_range():
    """End-to-end linear interpolation with value ranges."""
    print("=" * 60)
    print("Linear interpolation with value_ranges")
    print("=" * 60)

    data = torch.randn(100, 100, 100)
    torch.manual_seed(0)

    for N in [20_000, 100_000]:
        query = torch.rand(N, 3) * torch.tensor([1.0, 5.0, 10.0])

        for safety in [True, False]:
            dv = view.TorchMultidimView(
                data, value_ranges=[(0, 1), (0, 5), (0, 10)],
                check_safety=safety, method='linear',
            )
            med, std = bench(lambda: dv[query])
            label = "safe" if safety else "unsafe"
            print(f"  N={N:>7d}  {label:<6s}  {med:6.2f} ms (std {std:.2f})")
    print()


def bench_ravel_multi_index():
    """Isolate ravel_multi_index cost."""
    print("=" * 60)
    print("ravel_multi_index (standalone)")
    print("=" * 60)

    shape = (100, 100, 100)
    for N in [20_000, 100_000]:
        coords = torch.randint(0, 100, (N, 3))
        med, std = bench(lambda: view.ravel_multi_index(coords, shape))
        print(f"  N={N:>7d}  {med:6.2f} ms (std {std:.2f})")
    print()


def bench_ensure_index_key():
    """Isolate ensure_index_key cost."""
    print("=" * 60)
    print("ensure_index_key")
    print("=" * 60)

    data = torch.randn(100, 100, 100)
    dv = view.TorchMultidimView(data, value_ranges=[(0, 1), (0, 5), (0, 10)])

    for N in [20_000, 100_000]:
        key = torch.rand(N, 3) * torch.tensor([1.0, 5.0, 10.0])
        med, std = bench(lambda: dv.ensure_index_key(key))
        print(f"  N={N:>7d}  {med:6.2f} ms (std {std:.2f})")
    print()


def bench_get_valid_values():
    """Isolate get_valid_values cost."""
    print("=" * 60)
    print("get_valid_values")
    print("=" * 60)

    data = torch.randn(100, 100, 100)
    dv = view.TorchMultidimView(data, value_ranges=[(0, 1), (0, 5), (0, 10)])

    for N in [20_000, 100_000]:
        key = torch.rand(N, 3) * torch.tensor([1.0, 5.0, 10.0])
        med, std = bench(lambda: dv.get_valid_values(key))
        print(f"  N={N:>7d}  {med:6.2f} ms (std {std:.2f})")
    print()


def bench_get_valid_ravel_indices():
    """Isolate get_valid_ravel_indices cost (the full orchestration)."""
    print("=" * 60)
    print("get_valid_ravel_indices (full orchestration)")
    print("=" * 60)

    data = torch.randn(100, 100, 100)

    for N in [20_000, 100_000]:
        key = torch.rand(N, 3) * torch.tensor([1.0, 5.0, 10.0])

        for safety in [True, False]:
            dv = view.TorchMultidimView(
                data, value_ranges=[(0, 1), (0, 5), (0, 10)],
                check_safety=safety,
            )
            med, std = bench(lambda: dv.get_valid_ravel_indices(key))
            label = "safe" if safety else "unsafe"
            print(f"  N={N:>7d}  {label:<6s}  {med:6.2f} ms (std {std:.2f})")
    print()


def bench_numpy_nearest_value_range():
    """NumPy: end-to-end nearest-neighbor with value ranges."""
    print("=" * 60)
    print("[NumPy] Nearest-neighbor with value_ranges")
    print("=" * 60)

    np.random.seed(0)
    data = np.random.randn(100, 100, 100)

    for N in [20_000, 100_000]:
        query = np.random.rand(N, 3) * np.array([1.0, 5.0, 10.0])

        for safety in [True, False]:
            dv = np_view.NumpyMultidimView(
                data, value_ranges=[(0, 1), (0, 5), (0, 10)],
                check_safety=safety,
            )
            med, std = bench(lambda: dv[query])
            label = "safe" if safety else "unsafe"
            print(f"  N={N:>7d}  {label:<6s}  {med:6.2f} ms (std {std:.2f})")
    print()


def bench_numpy_linear_value_range():
    """NumPy: end-to-end linear interpolation with value ranges."""
    print("=" * 60)
    print("[NumPy] Linear interpolation with value_ranges")
    print("=" * 60)

    np.random.seed(0)
    data = np.random.randn(100, 100, 100)

    for N in [20_000, 100_000]:
        query = np.random.rand(N, 3) * np.array([1.0, 5.0, 10.0])

        for safety in [True, False]:
            dv = np_view.NumpyMultidimView(
                data, value_ranges=[(0, 1), (0, 5), (0, 10)],
                check_safety=safety, method='linear',
            )
            med, std = bench(lambda: dv[query])
            label = "safe" if safety else "unsafe"
            print(f"  N={N:>7d}  {label:<6s}  {med:6.2f} ms (std {std:.2f})")
    print()


if __name__ == "__main__":
    print(">>> PyTorch <<<\n")
    bench_ravel_multi_index()
    bench_ensure_index_key()
    bench_get_valid_values()
    bench_get_valid_ravel_indices()
    bench_nearest_value_range()
    bench_linear_value_range()
    print("\n>>> NumPy <<<\n")
    bench_numpy_nearest_value_range()
    bench_numpy_linear_value_range()
