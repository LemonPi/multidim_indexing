from multidim_indexing.view import MultidimView, classproperty
import numpy as np


class NumpyMultidimView(MultidimView):
    @classproperty
    def default_coordinate_dtype(cls):
        return float

    @classproperty
    def lib(cls):
        return np

    @property
    def int(self):
        return int

    def arr(self, *args, **kwargs):
        return np.array(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return np.zeros(*args, **kwargs)

    @classmethod
    def cast(cls, arr, dtype):
        return arr.astype(dtype)

    @classmethod
    def all(cls, arr, dim=0):
        return np.all(arr, axis=dim)

    @classmethod
    def any(cls, arr, dim=0):
        return np.any(arr, axis=dim)

    @classmethod
    def is_valid_arr_value(cls, val, valid):
        return not np.isscalar(val) and val.size == valid.size

    @classmethod
    def ravel_multi_index(cls, key, shape):
        if len(key.shape) == 2:
            key = key.T
        elif len(key.shape) == 3:
            key = key.transpose((1, 2, 0))
        return np.ravel_multi_index(tuple(key), shape)

    def unravel_key(self, key):
        return np.stack(np.unravel_index(key, self.shape)).T

    @classmethod
    def transpose(cls, arr):
        return np.transpose(arr)

    @classmethod
    def repeat(cls, arr, repeats):
        return np.tile(arr, repeats)

    @classmethod
    def cat(cls, arrs, dim=0):
        return np.concatenate(arrs, axis=dim)

    @classmethod
    def stack(cls, arrs, dim=0):
        return np.stack(arrs, axis=dim)
