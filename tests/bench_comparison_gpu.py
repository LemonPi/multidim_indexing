"""GPU benchmarks comparing multidim_indexing against torch alternatives.

Compares on CUDA:
  - TorchMultidimView (this library)
  - torch.nn.functional.grid_sample
  - torch advanced indexing (manual coordinate conversion)

Run: python tests/bench_comparison_gpu.py
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
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


def print_row(label, med, std, ref_med=None):
    speedup = f"  ({ref_med / med:.2f}x)" if ref_med is not None else ""
    print(f"  {label:<40s}  {med:8.2f} ms (std {std:.2f}){speedup}")


# ============================================================
# Setup: 3D grid (100^3) on GPU
# ============================================================
grid_size = 100
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 5, grid_size)
z = np.linspace(0, 10, grid_size)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
data_np = (2 * xx + 3 * yy - zz).astype(np.float32)
data_torch = torch.from_numpy(data_np).to(device)

ranges = [(0, 1), (0, 5), (0, 10)]


def run_nearest_benchmarks():
    print("=" * 70)
    print("[CUDA] NEAREST-NEIGHBOR LOOKUP (3D, 100^3 grid)")
    print("=" * 70)

    for N in [20_000, 100_000]:
        print(f"\n  N = {N:,}")
        print(f"  {'-' * 64}")

        torch.manual_seed(42)
        query = torch.rand(N, 3, device=device) * torch.tensor([1.0, 5.0, 10.0], device=device)

        # --- multidim_indexing ---
        mv = view.TorchMultidimView(data_torch, value_ranges=ranges, check_safety=False)
        med_ours, std_ours = bench(lambda: mv[query])

        mv_safe = view.TorchMultidimView(data_torch, value_ranges=ranges, check_safety=True)
        med_ours_safe, std_ours_safe = bench(lambda: mv_safe[query])

        # --- torch manual advanced indexing ---
        inv_res_t = torch.tensor([1.0 / ((1 - 0) / 99), 1.0 / ((5 - 0) / 99), 1.0 / ((10 - 0) / 99)],
                                 device=device)
        mins_t = torch.tensor([0.0, 0.0, 0.0], device=device)

        def torch_manual():
            idx = ((query - mins_t) * inv_res_t).round().long()
            idx = idx.clamp(0, grid_size - 1)
            return data_torch[idx[:, 0], idx[:, 1], idx[:, 2]]

        med_manual, std_manual = bench(torch_manual)

        print_row("multidim_indexing (unsafe)", med_ours, std_ours)
        print_row("multidim_indexing (safe)", med_ours_safe, std_ours_safe, med_ours)
        print_row("torch advanced indexing (manual)", med_manual, std_manual, med_ours)

    print()


def run_linear_benchmarks():
    print("=" * 70)
    print("[CUDA] LINEAR INTERPOLATION (3D, 100^3 grid)")
    print("=" * 70)

    for N in [20_000, 100_000]:
        print(f"\n  N = {N:,}")
        print(f"  {'-' * 64}")

        torch.manual_seed(42)
        query = torch.rand(N, 3, device=device) * torch.tensor([1.0, 5.0, 10.0], device=device)

        # --- multidim_indexing (linear) ---
        mv = view.TorchMultidimView(data_torch, value_ranges=ranges,
                                    check_safety=False, method='linear')
        med_ours, std_ours = bench(lambda: mv[query])

        mv_safe = view.TorchMultidimView(data_torch, value_ranges=ranges,
                                         check_safety=True, method='linear')
        med_ours_safe, std_ours_safe = bench(lambda: mv_safe[query])

        # --- torch.nn.functional.grid_sample ---
        gs_input = data_torch.unsqueeze(0).unsqueeze(0)  # (1, 1, 100, 100, 100)
        scale = torch.tensor([1.0, 5.0, 10.0], device=device)
        query_normalized = (query / scale) * 2 - 1  # [0, scale] -> [-1, 1]
        query_gs = query_normalized.flip(-1).reshape(1, 1, 1, N, 3)

        def grid_sample_call():
            return F.grid_sample(gs_input, query_gs, mode='bilinear',
                                 align_corners=True, padding_mode='zeros').reshape(-1)

        med_gs, std_gs = bench(grid_sample_call)

        print_row("multidim_indexing (unsafe)", med_ours, std_ours)
        print_row("multidim_indexing (safe)", med_ours_safe, std_ours_safe, med_ours)
        print_row("torch.nn.functional.grid_sample", med_gs, std_gs, med_ours)

    print()


if __name__ == "__main__":
    run_nearest_benchmarks()
    run_linear_benchmarks()
