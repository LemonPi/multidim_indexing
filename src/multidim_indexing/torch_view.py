import torch
from typing import List, Tuple, Union
from multidim_indexing.view import MultidimView, classproperty


class TorchMultidimView(MultidimView):
    def __init__(self, source, *args, **kwargs):
        self.device = source.device
        super(TorchMultidimView, self).__init__(source, *args, **kwargs)
        # Cache ravel stride coefficients — shape is fixed per instance
        shape_t = torch.tensor(self.shape + (1,), device=self.device)
        self._ravel_coefs = shape_t[1:].flipud().cumprod(dim=0).flipud()

    @classproperty
    def default_coordinate_dtype(cls):
        return torch.float32

    @classproperty
    def lib(cls):
        return torch

    @property
    def int(self):
        return torch.long

    def arr(self, *args, **kwargs):
        return torch.tensor(*args, device=self.device, **kwargs)

    def zeros(self, *args, **kwargs):
        return torch.zeros(*args, device=self.device, **kwargs)

    @classmethod
    def cast(cls, arr, dtype):
        return arr.to(dtype=dtype)

    @classmethod
    def all(cls, arr, dim=0):
        return torch.all(arr, dim=dim)

    @classmethod
    def any(cls, arr, dim=0):
        return torch.any(arr, dim=dim)

    @classmethod
    def is_valid_arr_value(cls, val, valid):
        return torch.is_tensor(val) and torch.numel(val) == torch.numel(valid)

    def ravel_multi_index(self, key, shape=None):
        return (key * self._ravel_coefs).sum(dim=-1)

    def unravel_key(self, key):
        return unravel_index(key, self.shape)

    def get_valid_ravel_indices(self, key):
        """Fused bounds-check, coordinate conversion, and ravel in one pass."""
        key = self._check_and_flatten_key(key)

        if self.check_safety:
            valid = (self._min <= key) & (key <= self._max)
            valid = torch.all(valid, dim=-1)
            valid_key = key[valid]
        else:
            valid = True
            valid_key = key

        if self._is_value_range and valid_key.dtype != self.int:
            index_key = ((valid_key - self._min) * self._inv_resolution).round().to(torch.long)
        else:
            index_key = valid_key

        flat_key = (index_key * self._ravel_coefs).sum(dim=-1)
        return flat_key, valid

    @classmethod
    def transpose(cls, arr):
        return arr.transpose(0, 1)

    @classmethod
    def repeat(cls, arr, repeats):
        return arr.repeat(*repeats)

    @classmethod
    def cat(cls, arrs, dim=0):
        return torch.cat(arrs, dim=dim)

    @classmethod
    def stack(cls, arrs, dim=0):
        return torch.stack(arrs, dim=dim)


# filling in functions from numpy from francois-rozet
Shape = Union[List[int], Tuple[int, ...], torch.Size]


def ravel_multi_index(coords: torch.Tensor, shape: Shape) -> torch.Tensor:
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.
    This is a `torch` implementation of `numpy.ravel_multi_index`.
    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.
    Returns:
        The raveled indices, (*,).
    """

    shape = coords.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs).sum(dim=-1)


def unravel_index(indices: torch.Tensor, shape: Shape) -> torch.Tensor:
    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.
    This is a `torch` implementation of `numpy.unravel_index`.
    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.
    Returns:
        The unraveled coordinates, (*, D).
    """

    shape = indices.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]


class BatchedViewLookup:
    """Vectorized lookup across multiple TorchMultidimView instances.

    Precomputes and stacks grid parameters from S views so that queries
    against all S views can be performed in a single vectorized pass
    instead of a Python loop.

    Usage::

        views = [TorchMultidimView(data_i, ranges_i, ...) for i in range(S)]
        batched = BatchedViewLookup(views)

        # pts: (S, N, D) — query points for each view, already in each view's coordinate frame
        values, valid = batched(pts)
        # values: (S, N) — looked-up values (invalid entries are 0)
        # valid:  (S, N) — boolean mask of in-bounds queries
    """

    def __init__(self, views: List[TorchMultidimView]):
        if not views:
            raise ValueError("views must be non-empty")

        device = views[0].device
        dtype = views[0].raw_data.dtype

        mins = []
        maxs = []
        inv_resolutions = []
        ravel_coefs_list = []
        data_offsets = [0]
        flat_data_parts = []

        for v in views:
            mins.append(v._min)
            maxs.append(v._max)
            inv_resolutions.append(v._inv_resolution)
            ravel_coefs_list.append(v._ravel_coefs)
            flat_data_parts.append(v.raw_data)
            data_offsets.append(data_offsets[-1] + v.raw_data.shape[0])

        self.mins = torch.stack(mins).to(device=device, dtype=dtype)                # (S, D)
        self.maxs = torch.stack(maxs).to(device=device, dtype=dtype)                # (S, D)
        self.inv_res = torch.stack(inv_resolutions).to(device=device, dtype=dtype)   # (S, D)
        self.ravel_coefs = torch.stack(ravel_coefs_list).to(device=device)           # (S, D)
        self.flat_data = torch.cat(flat_data_parts)                                  # (total_cells,)
        self.data_offsets = torch.tensor(data_offsets[:-1], device=device, dtype=torch.long)  # (S,)

    def __call__(self, pts: torch.Tensor):
        """Batched lookup across all S views.

        Args:
            pts: (S, N, D) query coordinates, one set per view.

        Returns:
            values: (S, N) looked-up values. Invalid entries are set to 0.
            valid: (S, N) boolean mask — True where the query was in bounds.
        """
        # 1. Bounds check: (S, N)
        valid = (pts >= self.mins[:, None, :]) & (pts <= self.maxs[:, None, :])
        valid = valid.all(dim=-1)

        # 2. Coordinate to index: (S, N, D)
        idx = ((pts - self.mins[:, None, :]) * self.inv_res[:, None, :]).round().long()

        # 3. Ravel to flat index: (S, N)
        flat_idx = (idx * self.ravel_coefs[:, None, :]).sum(dim=-1)
        # Clamp to avoid indexing errors on invalid entries (will be masked out)
        flat_idx = flat_idx.clamp(min=0)

        # 4. Offset into concatenated flat_data buffer: (S, N)
        global_idx = flat_idx + self.data_offsets[:, None]
        global_idx = global_idx.clamp(max=self.flat_data.shape[0] - 1)

        # 5. Single gather: (S, N)
        values = self.flat_data[global_idx]

        # Zero out invalid entries
        values[~valid] = 0

        return values, valid
