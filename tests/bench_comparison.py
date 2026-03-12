"""Benchmarks comparing multidim_indexing against alternative libraries.

Compares:
  - TorchMultidimView (this library)
  - scipy.interpolate.RegularGridInterpolator
  - torch.nn.functional.grid_sample
  - NumPy advanced indexing (manual coordinate conversion)

Run: python tests/bench_comparison.py
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator
from multidim_indexing import torch_view as view


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


def print_row(label, med, std, ref_med=None):
    speedup = f"  ({ref_med / med:.2f}x)" if ref_med is not None else ""
    print(f"  {label:<40s}  {med:8.2f} ms (std {std:.2f}){speedup}")


# ============================================================
# Setup: 3D grid (100^3), matching what CachedSDF uses
# ============================================================
grid_size = 100
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 5, grid_size)
z = np.linspace(0, 10, grid_size)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
data_np = (2 * xx + 3 * yy - zz).astype(np.float32)
data_torch = torch.from_numpy(data_np)

ranges = [(0, 1), (0, 5), (0, 10)]


def run_nearest_benchmarks():
    print("=" * 70)
    print("NEAREST-NEIGHBOR LOOKUP (3D, 100^3 grid)")
    print("=" * 70)

    for N in [20_000, 100_000]:
        print(f"\n  N = {N:,}")
        print(f"  {'-' * 64}")

        np.random.seed(42)
        query_np = np.random.rand(N, 3) * np.array([1.0, 5.0, 10.0])
        query_torch = torch.from_numpy(query_np).float()

        # --- multidim_indexing (this library) ---
        mv = view.TorchMultidimView(data_torch, value_ranges=ranges, check_safety=False)
        med_ours, std_ours = bench(lambda: mv[query_torch])

        mv_safe = view.TorchMultidimView(data_torch, value_ranges=ranges, check_safety=True)
        med_ours_safe, std_ours_safe = bench(lambda: mv_safe[query_torch])

        # --- scipy RegularGridInterpolator (nearest) ---
        rgi = RegularGridInterpolator((x, y, z), data_np, method='nearest',
                                      bounds_error=False, fill_value=-1)
        med_scipy, std_scipy = bench(lambda: rgi(query_np))

        # --- numpy advanced indexing (manual conversion) ---
        inv_res = 1.0 / np.array([(1 - 0) / 99, (5 - 0) / 99, (10 - 0) / 99])
        mins = np.array([0.0, 0.0, 0.0])

        def numpy_manual():
            idx = np.round((query_np - mins) * inv_res).astype(int)
            idx = np.clip(idx, 0, grid_size - 1)
            return data_np[idx[:, 0], idx[:, 1], idx[:, 2]]

        med_np, std_np = bench(numpy_manual)

        # --- torch manual advanced indexing ---
        inv_res_t = torch.from_numpy(inv_res).float()
        mins_t = torch.tensor([0.0, 0.0, 0.0])

        def torch_manual():
            idx = ((query_torch - mins_t) * inv_res_t).round().long()
            idx = idx.clamp(0, grid_size - 1)
            return data_torch[idx[:, 0], idx[:, 1], idx[:, 2]]

        med_torch_manual, std_torch_manual = bench(torch_manual)

        print_row("multidim_indexing (unsafe)", med_ours, std_ours)
        print_row("multidim_indexing (safe)", med_ours_safe, std_ours_safe, med_ours)
        print_row("scipy RegularGridInterpolator", med_scipy, std_scipy, med_ours)
        print_row("numpy advanced indexing (manual)", med_np, std_np, med_ours)
        print_row("torch advanced indexing (manual)", med_torch_manual, std_torch_manual, med_ours)

    print()


def run_linear_benchmarks():
    print("=" * 70)
    print("LINEAR INTERPOLATION (3D, 100^3 grid)")
    print("=" * 70)

    for N in [20_000, 100_000]:
        print(f"\n  N = {N:,}")
        print(f"  {'-' * 64}")

        np.random.seed(42)
        query_np = np.random.rand(N, 3) * np.array([1.0, 5.0, 10.0])
        query_torch = torch.from_numpy(query_np).float()

        # --- multidim_indexing (this library, linear) ---
        mv = view.TorchMultidimView(data_torch, value_ranges=ranges,
                                    check_safety=False, method='linear')
        med_ours, std_ours = bench(lambda: mv[query_torch])

        mv_safe = view.TorchMultidimView(data_torch, value_ranges=ranges,
                                         check_safety=True, method='linear')
        med_ours_safe, std_ours_safe = bench(lambda: mv_safe[query_torch])

        # --- scipy RegularGridInterpolator (linear) ---
        rgi = RegularGridInterpolator((x, y, z), data_np, method='linear',
                                      bounds_error=False, fill_value=-1)
        med_scipy, std_scipy = bench(lambda: rgi(query_np))

        # --- torch.nn.functional.grid_sample ---
        # grid_sample expects:
        #   input: (N, C, D, H, W) = (1, 1, 100, 100, 100)
        #   grid:  (N, D_out, H_out, W_out, 3) with coords in [-1, 1]
        # We reshape query (N, 3) -> (1, 1, 1, N, 3) and normalize to [-1, 1]
        gs_input = data_torch.unsqueeze(0).unsqueeze(0)  # (1, 1, 100, 100, 100)
        scale = torch.tensor([1.0, 5.0, 10.0])
        query_normalized = (query_torch / scale) * 2 - 1  # [0,1] -> [-1,1]
        # grid_sample uses (x,y,z) in reverse order relative to (D,H,W)
        query_gs = query_normalized.flip(-1)
        query_gs_shaped = query_gs.reshape(1, 1, 1, N, 3)

        def grid_sample_call():
            return F.grid_sample(gs_input, query_gs_shaped, mode='bilinear',
                                 align_corners=True, padding_mode='zeros').reshape(-1)

        med_gs, std_gs = bench(grid_sample_call)

        print_row("multidim_indexing (unsafe)", med_ours, std_ours)
        print_row("multidim_indexing (safe)", med_ours_safe, std_ours_safe, med_ours)
        print_row("scipy RegularGridInterpolator", med_scipy, std_scipy, med_ours)
        print_row("torch.nn.functional.grid_sample", med_gs, std_gs, med_ours)

    print()


def run_grid_sample_accuracy_check():
    """Verify grid_sample and multidim_indexing agree on linear interpolation."""
    print("=" * 70)
    print("ACCURACY CHECK: grid_sample vs multidim_indexing (linear)")
    print("=" * 70)

    N = 1000
    np.random.seed(42)
    query_np = np.random.rand(N, 3) * np.array([1.0, 5.0, 10.0])
    query_torch = torch.from_numpy(query_np).float()

    mv = view.TorchMultidimView(data_torch, value_ranges=ranges,
                                check_safety=True, method='linear')
    result_ours = mv[query_torch]

    gs_input = data_torch.unsqueeze(0).unsqueeze(0)
    scale = torch.tensor([1.0, 5.0, 10.0])
    query_normalized = (query_torch / scale) * 2 - 1
    query_gs = query_normalized.flip(-1).reshape(1, 1, 1, N, 3)
    result_gs = F.grid_sample(gs_input, query_gs, mode='bilinear',
                              align_corners=True, padding_mode='zeros').reshape(-1)

    # Ground truth for this linear function
    expected = 2 * query_torch[:, 0] + 3 * query_torch[:, 1] - query_torch[:, 2]

    err_ours = (result_ours - expected).abs().max().item()
    err_gs = (result_gs - expected).abs().max().item()
    print(f"  multidim_indexing max error: {err_ours:.6f}")
    print(f"  grid_sample max error:       {err_gs:.6f}")
    print()


if __name__ == "__main__":
    run_grid_sample_accuracy_check()
    run_nearest_benchmarks()
    run_linear_benchmarks()
