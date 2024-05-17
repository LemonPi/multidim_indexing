import abc
import itertools


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class MultidimView(abc.ABC):
    def __init__(self, source, value_ranges=None, invalid_value=-1, check_safety=True, method='nearest',
                 coord_dtype=None):
        """
        View into a tensor or numpy array that is convenient to index using a batch of indices.
        Intended for use on a cache of a function that needs to be indexed into with coordinates
        rather than integer indices.

        :param source: d dimensional underlying tensor or numpy array
        :param value_ranges: d pairs of [min, max] (or an otherwise sequence) coordinate keys for each of the
            d dimensions. It is an inclusive range, so indexing with the boundary values of either min or max is valid
        :param invalid_value: when check_safety is True, querying with out of range coordinates returns this value
            rather than raising an error. Needs to be the same dtype as source
        :param check_safety: whether to check if the keys are within bound; turn off to get about 40% speedup in
            indexing
        :param method: interpolation method when querying with value ranges, can be 'nearest' or 'linear'
        :param coord_dtype: dtype of the coordinate keys, defaults to float of the appropriate type
        """
        self.dtype = source.dtype
        self.shape = source.shape
        self.dim = len(source.shape)
        self.invalid_value = invalid_value
        self.check_safety = check_safety
        self.method = method
        self.coordinate_dtype = coord_dtype or self.default_coordinate_dtype

        if value_ranges is not None:
            self._min = self.arr([min(range) for range in value_ranges])
            self._max = self.arr([max(range) for range in value_ranges])
            self._is_value_range = True
            # want an inclusive range on the min and max, so indexing with max should be valid
            shape = self.arr(self.shape)
            self._resolution = (self._max - self._min) / (shape - 1)
            # if some dim of shape is 1, then resolution for it is undefined
            invalid_resolution = self.lib.isnan(self._resolution)
            valid_resolution_val = self._resolution[~invalid_resolution]
            if len(shape) != len(self._resolution):
                raise RuntimeError(f"Shape {shape} of source is not compatible with value_ranges {value_ranges}")
            if self.any(invalid_resolution) and len(valid_resolution_val) > 0:
                # we assume that it'll have the same resolution as the first non-1 dim
                self._resolution[invalid_resolution] = valid_resolution_val[0]
        else:
            self._min = self.lib.zeros(self.dim, dtype=self.int)
            self._max = self.arr(source.shape, dtype=self.int) - 1
            self._is_value_range = False

        # flattened view of the source data
        self._d = source.reshape(-1)

    @classproperty
    @abc.abstractmethod
    def default_coordinate_dtype(cls):
        """Return the default coordinate type for coordinate keys"""

    @classproperty
    @abc.abstractmethod
    def lib(cls):
        """Tensor/array library e.g. numpy and torch"""

    @property
    @abc.abstractmethod
    def int(self):
        """Return the integer type"""

    @property
    def raw_data(self):
        return self._d

    @abc.abstractmethod
    def arr(self, *args, **kwargs):
        """Create an array"""

    @abc.abstractmethod
    def zeros(self, *args, **kwargs):
        """Create a zero array"""

    @classmethod
    @abc.abstractmethod
    def cast(cls, arr, dtype):
        """Cast an arr into a dtype"""

    @classmethod
    @abc.abstractmethod
    def all(cls, arr, dim=0):
        """Logically evaluate to true iff all elements of arr is true"""

    @classmethod
    @abc.abstractmethod
    def any(cls, arr, dim=0):
        """Logically evaluate to true if any elements of arr is true"""

    @classmethod
    @abc.abstractmethod
    def is_valid_arr_value(cls, val, valid):
        """Evaluate if the value is an array of compatible size with valid mask"""

    @classmethod
    @abc.abstractmethod
    def ravel_multi_index(cls, key, shape):
        """Flatten keys from N x d to N"""

    @abc.abstractmethod
    def unravel_key(self, key):
        """Expand keys from N to N x d"""

    @classmethod
    @abc.abstractmethod
    def transpose(cls, arr):
        """Transpose dimensions 0 and 1"""

    @classmethod
    @abc.abstractmethod
    def repeat(cls, arr, repeats):
        """Repeat an array along its dimensions certain times"""

    @classmethod
    @abc.abstractmethod
    def cat(cls, arrs, dim=0):
        """Concatenate a sequence of arrays along a dimension"""

    @classmethod
    @abc.abstractmethod
    def stack(cls, arrs, dim=0):
        """Stack arrs along a newly created, specified dimension"""

    def is_key_ravelled(self, key):
        return len(key.shape) == 1 or (key.shape[-1] == 1 and self.dim != 1)

    def ensure_index_key(self, key, force=False):
        if self.is_key_ravelled(key):
            key = self.unravel_key(key.reshape(-1))

        if self._is_value_range and (force or key.dtype != self.int):
            offsets = (key - self._min) / self._resolution
            index_key = self.cast(offsets.round(), self.int)
            key = index_key

        return key

    def ensure_value_key(self, key, force=False):
        if self.is_key_ravelled(key):
            key = self.unravel_key(key.reshape(-1))
        # convert key from indices to value ranges if necessary
        if self._is_value_range and (force or key.dtype != self.coordinate_dtype):
            # scales = torch.tensor(self._resolution, device=key.device, dtype=self.coordinate_dtype)
            scales = self._resolution
            # offsets = torch.tensor(self._min, device=key.device, dtype=self.coordinate_dtype)
            offsets = self.cast(self._min, self.coordinate_dtype)

            # Broadcasting the scales and offsets across the last dimension
            value_key = key * scales + offsets

            if key.dtype != self.coordinate_dtype:
                value_key = value_key.to(self.coordinate_dtype)

            key = value_key

        return key

    def get_valid_values(self, key):
        is_valid = (self._min <= key) & (key <= self._max)
        return self.all(is_valid, dim=-1)

    def _check_and_flatten_key(self, key):
        """Flatten batch dimensions of key to ensure it is N x d"""
        # check if shorthand is used where batch indices are not specified
        if key.shape[-1] == self.dim - 1:
            B = key.shape[0]
            batch_index = self.repeat(self.lib.arange(B).reshape(B, 1, 1), (1, key.shape[-2], 1))
            key = self.cat((batch_index, key), dim=-1)

        # flatten batch dimensions
        key = key.reshape(-1, key.shape[-1])
        return key

    def get_valid_ravel_indices(self, key):
        """
        Ravel a N x d key into a N length key of ravelled indices
        :param key: N x d key (could be values or indices depending on underlying data)
        :return: ravelled indices of length N along with boolean validity mask of length N
        """
        key = self._check_and_flatten_key(key)

        # eliminate keys outside query
        if self.check_safety:
            valid = self.get_valid_values(key)
            key = key[valid]
        else:
            valid = True
        key = self.ensure_index_key(key)
        # flatten
        flat_key = self.ravel_multi_index(key, self.shape)
        return flat_key, valid

    def __getitem__(self, key):
        """
        Get a batch (size N) from the data with dimension d, indexed by a multidimensional key
        :param key: N x d query, with each row corresponding to one element to look up
        :return:
        """
        orig_key_shape = key.shape

        if self.method == 'nearest' or not self._is_value_range:
            flat_key, valid = self.get_valid_ravel_indices(key)
            if self.check_safety:
                N = valid.shape[0]
                res = self.zeros(N, dtype=self.dtype)
                res[valid] = self._d[flat_key]
                if callable(self.invalid_value):
                    invalid_entries = ~valid.reshape(key.shape[:-1])
                    res[~valid] = self.invalid_value(key[invalid_entries])
                else:
                    res[~valid] = self.invalid_value
            else:
                res = self._d[flat_key]
        elif self.method == 'linear':
            key = self._check_and_flatten_key(key)
            idx_raw = self.stack([(key[..., i] - self._min[i]) / self._resolution[i] for i in range(self.dim)], dim=-1)
            idx_left = self.cast(self.lib.floor(idx_raw), self.int)
            idx_right = idx_left + 1

            # check validity directly on the bounds of the indices
            a = self.lib.logical_and(idx_left >= 0, idx_left < self.arr(self.shape, dtype=self.int))
            b = self.lib.logical_and(idx_right >= 0, idx_right < self.arr(self.shape, dtype=self.int))
            valid = self.all(a & b, dim=-1)
            idx_left = idx_left[valid]
            idx_right = idx_right[valid]

            idxs = list(zip(idx_left.T, idx_right.T))

            dists_left = idx_raw[valid] - idx_left
            dists_right = 1 - dists_left
            dists = list(zip(dists_left.T, dists_right.T))

            # iterate over the vertices of a hypercube
            values = 0
            for indexer in itertools.product([0, 1], repeat=self.dim):
                this_idx = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
                flat_idx = self.ravel_multi_index(self.stack(this_idx, dim=-1), self.shape)
                deltas = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
                values = values + self._d[flat_idx] * self.lib.prod(self.stack(deltas), dim=0)

            if self.check_safety:
                N = valid.shape[0]
                res = self.zeros(N, dtype=self.dtype)
                res[valid] = values
                if callable(self.invalid_value):
                    invalid_entries = ~valid.reshape(idx_raw.shape[:-1])
                    res[~valid] = self.invalid_value(idx_raw[invalid_entries])
                else:
                    res[~valid] = self.invalid_value
            else:
                res = values
        else:
            raise NotImplementedError('method {} not implemented'.format(self.method))

        return res.reshape(list(orig_key_shape[:-2]) + [-1])

    def __setitem__(self, key, value):
        """
        Batch (size N) assignment
        :param key: N x d query, with each row corresponding to one element to look up
        :param value: value of compatible type and shape
        :return:
        """
        flat_key, valid = self.get_valid_ravel_indices(key)
        if self.is_valid_arr_value(value, valid):
            value = value.reshape(-1)
            if self.check_safety:
                value = value[valid]
        self._d[flat_key] = value
