import abc


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class MultidimView(abc.ABC):
    def __init__(self, source, value_ranges=None, invalid_value=-1, check_safety=True):
        """
        View into a tensor or numpy array that is convenient to index using a batch of indices.
        Intended for use on a cache of a function that needs to be indexed into with coordinates
        rather than integer indices.

        :param source: d dimensional underlying tensor or numpy array
        :param value_ranges: d pairs of [min, max] coordinate keys for each of the d dimensions. It is an inclusive
            range, so indexing with the boundary values of either min or max is valid
        :param invalid_value: when check_safety is True, querying with out of range coordinates returns this value
            rather than raising an error. Needs to be the same dtype as source
        :param check_safety: whether to check if the keys are within bound; turn off to get about 40% speedup in
            indexing
        """
        self.dtype = source.dtype
        self.shape = source.shape
        self.dim = len(source.shape)
        self.invalid_value = invalid_value
        self.check_safety = check_safety

        if value_ranges is not None:
            self._min = self.arr([range[0] for range in value_ranges])
            self._max = self.arr([range[1] for range in value_ranges])
            self._is_value_range = True
            # want an inclusive range on the min and max, so indexing with max should be valid
            self._resolution = (self._max - self._min) / (self.arr(self.shape) - 1)
        else:
            self._min = self.lib.zeros(self.dim, dtype=self.int)
            self._max = self.arr(source.shape, dtype=self.int) - 1
            self._is_value_range = False

        # flattened view of the source data
        self._d = source.reshape(-1)

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
        # convert key from value ranges to indices if necessary
        if self._is_value_range and (force or key.dtype != self.int):
            index_key = self.stack(
                [self.cast(self.lib.round((key[..., i] - self._min[i]) / self._resolution[i]), self.int) for i in
                 range(self.dim)],
                dim=-1)
            key = index_key
        return key

    def ensure_value_key(self, key, force=False):
        if self.is_key_ravelled(key):
            key = self.unravel_key(key.reshape(-1))
        # convert key from indices to value ranges if necessary
        if self._is_value_range and (force or key.dtype != self.dtype):
            value_key = self.stack(
                [self.cast(key[..., i] * self._resolution[i] + self._min[i], self.dtype) for i in range(self.dim)],
                dim=-1)
            key = value_key
        return key

    def get_valid_values(self, key):
        return self.all(
            self.stack([(self._min[i] <= key[..., i]) & (key[..., i] <= self._max[i]) for i in range(self.dim)]),
            dim=0)

    def get_valid_ravel_indices(self, key):
        """
        Ravel a N x d key into a N length key of ravelled indices
        :param key: N x d key (could be values or indices depending on underlying data)
        :return: ravelled indices of length N along with boolean validity mask of length N
        """
        # check if shorthand is used where batch indices are not specified
        if key.shape[-1] == self.dim - 1:
            B = key.shape[0]
            batch_index = self.repeat(self.lib.arange(B).reshape(B, 1, 1), (1, key.shape[-2], 1))
            key = self.cat((batch_index, key), dim=-1)

        # flatten batch dimensions
        key = key.reshape(-1, key.shape[-1])

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
