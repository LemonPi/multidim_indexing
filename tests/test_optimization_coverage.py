"""Tests covering gaps identified before optimization work.

These tests exercise code paths that the existing tests miss but that
the planned optimizations (see OPTIMIZE.md) will touch:
  - Nearest-neighbor with value_ranges + check_safety=False
  - Linear interpolation without scipy dependency (manual ground truth)
  - Linear interpolation with check_safety=False
  - Linear interpolation in 2D and 4D (existing test only covers 3D)
  - Single-dimension shapes with value ranges (infinite resolution edge case)

Both PyTorch and NumPy backends are tested where applicable.
"""

import torch
import numpy as np
from multidim_indexing import torch_view as view
from multidim_indexing import numpy_view as np_view


def test_nearest_value_range_no_safety():
    """Nearest-neighbor query with value_ranges and check_safety=False.

    This is the exact hot path from OPTIMIZE.md (CachedSDF usage) that
    had no dedicated correctness test.
    """
    shape = (10, 20, 30)
    high = 10 * 20 * 30
    data = torch.arange(0, high, dtype=torch.float).reshape(shape)
    data_view = view.TorchMultidimView(
        data, value_ranges=[(0, 1), (0, 2), (0, 3)], check_safety=False
    )
    data_view_safe = view.TorchMultidimView(
        data, value_ranges=[(0, 1), (0, 2), (0, 3)], check_safety=True
    )

    # Generate valid keys within range
    N = 500
    key = torch.rand(N, 3) * torch.tensor([1.0, 2.0, 3.0])
    result_unsafe = data_view[key]
    result_safe = data_view_safe[key]

    # For valid keys, safe and unsafe should agree
    # safe marks out-of-range as invalid_value; since all keys are in range, results match
    assert torch.allclose(result_unsafe, result_safe)


def test_nearest_value_range_no_safety_batch():
    """Batch query variant of the value-range no-safety path."""
    shape = (10, 20)
    data = torch.arange(0, 200, dtype=torch.float).reshape(shape)
    data_view = view.TorchMultidimView(
        data, value_ranges=[(0, 1), (0, 5)], check_safety=False
    )
    data_view_safe = view.TorchMultidimView(
        data, value_ranges=[(0, 1), (0, 5)], check_safety=True
    )

    B, N = 4, 100
    key = torch.rand(B, N, 2) * torch.tensor([1.0, 5.0])
    result_unsafe = data_view[key]
    result_safe = data_view_safe[key]
    assert result_unsafe.shape == (B, N)
    assert torch.allclose(result_unsafe, result_safe)


def test_linear_interpolation_2d():
    """Linear interpolation on a 2D grid with manually computed ground truth.

    Uses f(x, y) = 3x + 7y on a grid, so interpolation is exact for any
    point in the domain.
    """
    x = np.linspace(0, 4, 9)
    y = np.linspace(0, 6, 13)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    values = 3 * xx + 7 * yy

    m = view.TorchMultidimView(
        torch.tensor(values), [x, y], invalid_value=0, method='linear'
    )

    N = 200
    torch.manual_seed(42)
    query = torch.rand(N, 2) * torch.tensor([4.0, 6.0])
    result = m[query]
    expected = 3 * query[:, 0] + 7 * query[:, 1]
    assert torch.allclose(result, expected.to(result.dtype), atol=1e-5)


def test_linear_interpolation_3d():
    """Linear interpolation on 3D grid, no scipy needed."""
    x = np.linspace(0, 4, 5)
    y = np.linspace(0, 5, 6)
    z = np.linspace(0, 6, 7)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    values = 2 * xx + 3 * yy - zz

    m = view.TorchMultidimView(
        torch.tensor(values), [x, y, z], invalid_value=0, method='linear'
    )

    N = 500
    torch.manual_seed(42)
    query = torch.rand(N, 3) * torch.tensor([4.0, 5.0, 6.0])
    result = m[query]
    expected = 2 * query[:, 0] + 3 * query[:, 1] - query[:, 2]
    assert torch.allclose(result, expected.to(result.dtype), atol=1e-4)


def test_linear_interpolation_4d():
    """Linear interpolation on 4D grid."""
    x = np.linspace(0, 2, 5)
    y = np.linspace(0, 3, 6)
    z = np.linspace(0, 4, 7)
    w = np.linspace(0, 5, 8)
    xx, yy, zz, ww = np.meshgrid(x, y, z, w, indexing='ij')
    values = xx + 2 * yy + 3 * zz + 4 * ww

    m = view.TorchMultidimView(
        torch.tensor(values), [x, y, z, w], invalid_value=0, method='linear'
    )

    N = 200
    torch.manual_seed(42)
    query = torch.rand(N, 4) * torch.tensor([2.0, 3.0, 4.0, 5.0])
    result = m[query]
    expected = query[:, 0] + 2 * query[:, 1] + 3 * query[:, 2] + 4 * query[:, 3]
    assert torch.allclose(result, expected.to(result.dtype), atol=1e-3)


def test_linear_interpolation_no_safety():
    """Linear interpolation with check_safety=False, valid keys only."""
    x = np.linspace(0, 4, 9)
    y = np.linspace(0, 6, 13)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    values = 3 * xx + 7 * yy

    m_safe = view.TorchMultidimView(
        torch.tensor(values), [x, y], invalid_value=0, method='linear',
        check_safety=True,
    )
    m_unsafe = view.TorchMultidimView(
        torch.tensor(values), [x, y], invalid_value=0, method='linear',
        check_safety=False,
    )

    N = 200
    torch.manual_seed(42)
    query = torch.rand(N, 2) * torch.tensor([4.0, 6.0])
    result_safe = m_safe[query]
    result_unsafe = m_unsafe[query]
    assert torch.allclose(result_safe, result_unsafe)


def test_single_dim_shape_with_value_ranges():
    """Grid with a size-1 dimension (triggers infinite resolution fix d9edf77)."""
    shape = (1, 50, 50)
    data = torch.arange(0, 2500, dtype=torch.float).reshape(shape)
    data_view = view.TorchMultidimView(
        data, value_ranges=[(0, 0), (0, 1), (0, 1)], check_safety=True
    )

    N = 100
    torch.manual_seed(42)
    key = torch.zeros(N, 3)
    key[:, 0] = 0.0  # only valid value for the size-1 dim
    key[:, 1] = torch.rand(N)
    key[:, 2] = torch.rand(N)
    result = data_view[key]
    assert result.shape == (N,)
    # No NaN or all-invalid
    assert not torch.all(result == data_view.invalid_value)


# ==================== BatchedViewLookup tests ====================


def test_batched_lookup_matches_sequential():
    """BatchedViewLookup should match querying each view individually."""
    S = 4
    N = 500
    torch.manual_seed(42)

    views = []
    for i in range(S):
        data = torch.randn(10 + i, 20 + i, 30 + i)
        ranges = [(0, 1.0 + i), (0, 2.0 + i), (0, 3.0 + i)]
        views.append(view.TorchMultidimView(data, value_ranges=ranges, check_safety=True))

    batched = view.BatchedViewLookup(views)

    # Generate in-range queries per view
    pts_list = []
    for i, v in enumerate(views):
        q = torch.rand(N, 3) * torch.tensor([1.0 + i, 2.0 + i, 3.0 + i])
        pts_list.append(q)
    pts = torch.stack(pts_list)  # (S, N, 3)

    values, valid = batched(pts)
    assert values.shape == (S, N)
    assert valid.shape == (S, N)

    # Compare against sequential single-view queries
    for i, v in enumerate(views):
        expected = v[pts[i]]
        # All points are in range, so all should be valid
        assert torch.all(valid[i])
        assert torch.allclose(values[i], expected)


def test_batched_lookup_out_of_bounds():
    """BatchedViewLookup returns valid=False for out-of-bounds queries."""
    data = torch.randn(10, 10, 10)
    views = [
        view.TorchMultidimView(data, value_ranges=[(0, 1), (0, 1), (0, 1)]),
        view.TorchMultidimView(data, value_ranges=[(0, 1), (0, 1), (0, 1)]),
    ]
    batched = view.BatchedViewLookup(views)

    N = 100
    torch.manual_seed(42)
    # Mix of in-bounds and out-of-bounds
    pts_in = torch.rand(N // 2, 3)
    pts_out = torch.rand(N // 2, 3) + 5.0  # way out of range
    pts_mixed = torch.cat([pts_in, pts_out], dim=0)
    pts = torch.stack([pts_mixed, pts_mixed])  # (2, N, 3)

    values, valid = batched(pts)
    assert valid[:, :N // 2].all()
    assert not valid[:, N // 2:].any()
    assert (values[~valid] == 0).all()


def test_batched_lookup_different_shapes():
    """BatchedViewLookup works with views of different grid shapes."""
    torch.manual_seed(42)
    views = [
        view.TorchMultidimView(torch.randn(5, 10, 15), value_ranges=[(0, 1), (0, 2), (0, 3)]),
        view.TorchMultidimView(torch.randn(20, 25, 30), value_ranges=[(0, 1), (0, 2), (0, 3)]),
    ]
    batched = view.BatchedViewLookup(views)

    N = 200
    pts = torch.rand(2, N, 3) * torch.tensor([1.0, 2.0, 3.0])
    values, valid = batched(pts)

    for i, v in enumerate(views):
        expected = v[pts[i]]
        in_bounds = valid[i]
        assert torch.allclose(values[i][in_bounds], expected[in_bounds])


# ==================== NumPy backend tests ====================


def test_numpy_nearest_value_range_no_safety():
    """NumPy: nearest-neighbor with value_ranges and check_safety=False."""
    shape = (10, 20, 30)
    high = 10 * 20 * 30
    data = np.arange(0, high, dtype=float).reshape(shape)
    data_view = np_view.NumpyMultidimView(
        data, value_ranges=[(0, 1), (0, 2), (0, 3)], check_safety=False
    )
    data_view_safe = np_view.NumpyMultidimView(
        data, value_ranges=[(0, 1), (0, 2), (0, 3)], check_safety=True
    )

    N = 500
    np.random.seed(42)
    key = np.random.rand(N, 3) * np.array([1.0, 2.0, 3.0])
    result_unsafe = data_view[key]
    result_safe = data_view_safe[key]
    assert np.allclose(result_unsafe, result_safe)


def test_numpy_nearest_value_range_no_safety_batch():
    """NumPy: batch query with value_ranges and check_safety=False."""
    shape = (10, 20)
    data = np.arange(0, 200, dtype=float).reshape(shape)
    data_view = np_view.NumpyMultidimView(
        data, value_ranges=[(0, 1), (0, 5)], check_safety=False
    )
    data_view_safe = np_view.NumpyMultidimView(
        data, value_ranges=[(0, 1), (0, 5)], check_safety=True
    )

    B, N = 4, 100
    np.random.seed(42)
    key = np.random.rand(B, N, 2) * np.array([1.0, 5.0])
    result_unsafe = data_view[key]
    result_safe = data_view_safe[key]
    assert result_unsafe.shape == (B, N)
    assert np.allclose(result_unsafe, result_safe)


def test_numpy_linear_interpolation_2d():
    """NumPy: linear interpolation on 2D grid."""
    x = np.linspace(0, 4, 9)
    y = np.linspace(0, 6, 13)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    values = 3 * xx + 7 * yy

    m = np_view.NumpyMultidimView(
        values, [x, y], invalid_value=0, method='linear'
    )

    N = 200
    np.random.seed(42)
    query = np.random.rand(N, 2) * np.array([4.0, 6.0])
    result = m[query]
    expected = 3 * query[:, 0] + 7 * query[:, 1]
    assert np.allclose(result, expected, atol=1e-5)


def test_numpy_linear_interpolation_3d():
    """NumPy: linear interpolation on 3D grid."""
    x = np.linspace(0, 4, 5)
    y = np.linspace(0, 5, 6)
    z = np.linspace(0, 6, 7)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    values = 2 * xx + 3 * yy - zz

    m = np_view.NumpyMultidimView(
        values, [x, y, z], invalid_value=0, method='linear'
    )

    N = 500
    np.random.seed(42)
    query = np.random.rand(N, 3) * np.array([4.0, 5.0, 6.0])
    result = m[query]
    expected = 2 * query[:, 0] + 3 * query[:, 1] - query[:, 2]
    assert np.allclose(result, expected, atol=1e-4)


def test_numpy_single_dim_shape_with_value_ranges():
    """NumPy: grid with a size-1 dimension."""
    shape = (1, 50, 50)
    data = np.arange(0, 2500, dtype=float).reshape(shape)
    data_view = np_view.NumpyMultidimView(
        data, value_ranges=[(0, 0), (0, 1), (0, 1)], check_safety=True
    )

    N = 100
    np.random.seed(42)
    key = np.zeros((N, 3))
    key[:, 0] = 0.0
    key[:, 1] = np.random.rand(N)
    key[:, 2] = np.random.rand(N)
    result = data_view[key]
    assert result.shape == (N,)
    assert not np.all(result == data_view.invalid_value)


if __name__ == "__main__":
    # Torch tests
    test_nearest_value_range_no_safety()
    test_nearest_value_range_no_safety_batch()
    test_linear_interpolation_2d()
    test_linear_interpolation_3d()
    test_linear_interpolation_4d()
    test_linear_interpolation_no_safety()
    test_single_dim_shape_with_value_ranges()
    # NumPy tests
    test_numpy_nearest_value_range_no_safety()
    test_numpy_nearest_value_range_no_safety_batch()
    test_numpy_linear_interpolation_2d()
    test_numpy_linear_interpolation_3d()
    test_numpy_single_dim_shape_with_value_ranges()
    print("All tests passed.")
