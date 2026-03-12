"""CUDA GPU benchmarks for comparison with CPU PyTorch and NumPy."""

import time
import torch
import numpy as np
from multidim_indexing import torch_view as view

assert torch.cuda.is_available(), "CUDA not available"
device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}\n")


def bench(fn, warmup=5, runs=20):
    """Run fn with CUDA sync, return median time in ms."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times)
    return np.median(times), np.std(times)


def bench_nearest_value_range():
    print("=" * 60)
    print("[CUDA] Nearest-neighbor with value_ranges")
    print("=" * 60)

    data = torch.randn(100, 100, 100, device=device)
    torch.manual_seed(0)

    for N in [20_000, 100_000]:
        query = (torch.rand(N, 3, device=device)
                 * torch.tensor([1.0, 5.0, 10.0], device=device))

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
    print("=" * 60)
    print("[CUDA] Linear interpolation with value_ranges")
    print("=" * 60)

    data = torch.randn(100, 100, 100, device=device)
    torch.manual_seed(0)

    for N in [20_000, 100_000]:
        query = (torch.rand(N, 3, device=device)
                 * torch.tensor([1.0, 5.0, 10.0], device=device))

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
    print("=" * 60)
    print("[CUDA] ravel_multi_index (standalone)")
    print("=" * 60)

    shape = (100, 100, 100)
    for N in [20_000, 100_000]:
        coords = torch.randint(0, 100, (N, 3), device=device)
        med, std = bench(lambda: view.ravel_multi_index(coords, shape))
        print(f"  N={N:>7d}  {med:6.2f} ms (std {std:.2f})")
    print()


def bench_get_valid_ravel_indices():
    print("=" * 60)
    print("[CUDA] get_valid_ravel_indices (full orchestration)")
    print("=" * 60)

    data = torch.randn(100, 100, 100, device=device)

    for N in [20_000, 100_000]:
        query = (torch.rand(N, 3, device=device)
                 * torch.tensor([1.0, 5.0, 10.0], device=device))

        for safety in [True, False]:
            dv = view.TorchMultidimView(
                data, value_ranges=[(0, 1), (0, 5), (0, 10)],
                check_safety=safety,
            )
            med, std = bench(lambda: dv.get_valid_ravel_indices(query))
            label = "safe" if safety else "unsafe"
            print(f"  N={N:>7d}  {label:<6s}  {med:6.2f} ms (std {std:.2f})")
    print()


if __name__ == "__main__":
    bench_ravel_multi_index()
    bench_get_valid_ravel_indices()
    bench_nearest_value_range()
    bench_linear_value_range()
