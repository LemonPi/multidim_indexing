import torch
from typing import List, Tuple, Union
from multidim_indexing.view import MultidimView, classproperty


class TorchMultidimView(MultidimView):
    def __init__(self, source, *args, **kwargs):
        self.device = source.device
        super(TorchMultidimView, self).__init__(source, *args, **kwargs)

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

    @classmethod
    def ravel_multi_index(cls, key, shape):
        return ravel_multi_index(key, shape)

    def unravel_key(self, key):
        return unravel_index(key, self.shape)

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
