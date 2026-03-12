# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
pip install -e ".[all,test]"        # Install with all deps (torch + numpy + pytest)
pytest tests/                        # Run all tests
pytest tests/test_torch_view.py      # PyTorch tests only
pytest tests/test_numpy_view.py      # NumPy tests only
pytest tests/test_torch_view.py -k "test_batch_index"  # Single test
python tests/test_performance.py     # Performance benchmarks (100³ grid, 20k-200k queries)
```

No linter or formatter is configured.

## Architecture

Python library for batch multidimensional indexing of PyTorch tensors and NumPy arrays. The primary use case is querying large pre-computed grids (e.g., signed distance fields) with batches of coordinate points.

### Class Hierarchy

`MultidimView` (abstract base in `view.py`) → `TorchMultidimView` (`torch_view.py`) / `NumpyMultidimView` (`numpy_view.py`)

The base class uses the Template Method pattern: it defines the indexing/interpolation logic while subclasses implement library-specific array operations (`lib`, `arr`, `zeros`, `cast`, `ravel_multi_index`, etc.).

### Core Data Model

On construction, the source tensor is **flattened to 1D** (`self._d = source.reshape(-1)`). All queries convert N×d coordinate batches into flat indices via `ravel_multi_index`, then do a single vectorized lookup on the flat array.

Value ranges map real-world coordinates to grid indices: `idx = (coord - min) / resolution`.

### Hot Paths (Performance-Critical)

**Nearest-neighbor `__getitem__`:** `_check_and_flatten_key` → `get_valid_ravel_indices` (converts coordinates to flat indices) → `self._d[flat_key]` → reshape.

**Linear interpolation `__getitem__`** (the main bottleneck for large batches):
1. Compute fractional grid indices from coordinates
2. Generate all 2^d corner combinations via Cartesian product of `[floor, ceil]`
3. **Batch ravel all N×2^d corner indices at once** (key 3x optimization from commit `5ec2e52`)
4. Compute interpolation weights as N×2^d matrix using `weight = prod(frac^offset * (1-frac)^(1-offset))`
5. Single vectorized lookup + weighted sum: `(self._d[flat_corners] * weights).sum(dim=1)`

**Safety vs speed:** `check_safety=True` (default) adds validity masking at ~40% overhead. `check_safety=False` skips all bounds checking.

**`__setitem__`:** Same flat-index path, then direct assignment `self._d[flat_key] = value`.

### torch_view.py Helper Functions

`ravel_multi_index` and `unravel_index` are custom implementations (PyTorch doesn't provide these natively). These are called on every query.

