import torch
import numpy as np
from typing import List, Tuple, Union


class View:
    def __init__(self, source, value_ranges=None, invalid_value=-1):
        self.dtype = source.dtype
        self.device = source.device
        self.shape = source.shape
        self.dim = len(source.shape)
        self.invalid_value = invalid_value

        if value_ranges is not None:
            self._min = torch.tensor([range[0] for range in value_ranges], device=self.device, dtype=self.dtype)
            self._max = torch.tensor([range[0] for range in value_ranges], device=self.device, dtype=self.dtype)
            self._is_value_range = True
            self._resolution = (self._max - self._min) / self.shape
        else:
            self._min = torch.zeros(self.dim, device=self.device, dtype=torch.long)
            self._max = torch.tensor(source.shape, device=self.device, dtype=torch.long) - 1
            self._is_value_range = False

        # flattened view of the source data
        self._d = source.view(-1)

    def get_valid_ravel_indices(self, key):
        """
        Ravel a N x d key into a N length key of ravelled indices
        :param key: N x d key (could be values or indices depending on underlying data)
        :return: ravelled indices of length N along with boolean validity mask of length N
        """
        # convert key from value ranges to indices if necessary
        if self._is_value_range:
            index_key = (torch.round((key[:, i] - self._min[i]) / self._resolution[i]).to(dtype=torch.long) for i in
                         range(self.dim))
            key = index_key

        # eliminate keys outside query
        valid = torch.all(
            torch.stack([(self._min[i] <= key[:, i]) & (key[:, i] <= self._max[i]) for i in range(self.dim)]), dim=0)
        key = key[valid]
        # flatten
        flat_key = ravel_multi_index(key, self.shape)
        return flat_key, valid

    def __getitem__(self, key):
        """
        Get a batch (size N) from the data with dimension d, indexed by a multidimensional key
        :param key: N x d query, with each row corresponding to one element to look up
        :return:
        """

        flat_key, valid = self.get_valid_ravel_indices(key)
        N = key.shape[0]
        res = torch.ones(N, dtype=self.dtype, device=self.device) * self.invalid_value
        res[valid] = self._d[flat_key]
        return res


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
