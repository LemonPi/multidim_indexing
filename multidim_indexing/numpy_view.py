from multidim_indexing.view import View, classproperty
import numpy as np


class NumpyView(View):
    @classproperty
    def lib(cls):
        return np

    def arr(self, *args, **kwargs):
        return np.array(*args, **kwargs)

    @classmethod
    def cast(cls, arr, dtype):
        return arr.astype(dtype)

    @classmethod
    def all(cls, arr, dim=0):
        return np.all(arr, axis=dim)

    @classmethod
    def is_valid_arr_value(cls, val, valid):
        return not np.isscalar(val) and val.size == valid.size

    @classmethod
    def ravel_multi_index(cls, key, shape):
        return np.ravel_multi_index(tuple(key.T), shape)

    @classmethod
    def transpose(cls, arr):
        return np.transpose(arr)
