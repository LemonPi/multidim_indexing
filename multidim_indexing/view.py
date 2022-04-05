import abc


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class View(abc.ABC):
    def __init__(self, source, value_ranges=None, invalid_value=-1):
        self.dtype = source.dtype
        self.shape = source.shape
        self.dim = len(source.shape)
        self.invalid_value = invalid_value

        if value_ranges is not None:
            self._min = self.arr([range[0] for range in value_ranges], dtype=self.dtype)
            self._max = self.arr([range[1] for range in value_ranges], dtype=self.dtype)
            self._is_value_range = True
            self._resolution = (self._max - self._min) / self.arr(self.shape)
        else:
            self._min = self.lib.zeros(self.dim, dtype=self.lib.long)
            self._max = self.arr(source.shape, dtype=self.lib.long) - 1
            self._is_value_range = False

        # flattened view of the source data
        self._d = source.reshape(-1)

    @classproperty
    @abc.abstractmethod
    def lib(cls):
        """Tensor/array library e.g. numpy and torch"""

    @abc.abstractmethod
    def arr(self, *args, **kwargs):
        """Create an array"""

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
        """flatten keys from N x d to N"""

    @classmethod
    @abc.abstractmethod
    def transpose(cls, arr):
        """Transpose dimensions 0 and 1"""

    def get_valid_ravel_indices(self, key):
        """
        Ravel a N x d key into a N length key of ravelled indices
        :param key: N x d key (could be values or indices depending on underlying data)
        :return: ravelled indices of length N along with boolean validity mask of length N
        """
        # eliminate keys outside query
        valid = self.all(
            self.lib.stack([(self._min[i] <= key[:, i]) & (key[:, i] <= self._max[i]) for i in range(self.dim)]), dim=0)
        key = key[valid]
        # convert key from value ranges to indices if necessary
        if self._is_value_range:
            index_key = self.transpose(self.lib.stack(
                [self.cast(self.lib.round((key[:, i] - self._min[i]) / self._resolution[i]), self.lib.long) for i in
                 range(self.dim)]))
            key = index_key

        # flatten
        flat_key = self.ravel_multi_index(key, self.shape)
        return flat_key, valid

    def __getitem__(self, key):
        """
        Get a batch (size N) from the data with dimension d, indexed by a multidimensional key
        :param key: N x d query, with each row corresponding to one element to look up
        :return:
        """

        flat_key, valid = self.get_valid_ravel_indices(key)
        N = key.shape[0]
        res = self.lib.ones(N, dtype=self.dtype) * self.invalid_value
        res[valid] = self._d[flat_key]
        return res

    def __setitem__(self, key, value):
        """
        Batch (size N) assignment
        :param key: N x d query, with each row corresponding to one element to look up
        :param value: value of compatible type and shape
        :return:
        """
        flat_key, valid = self.get_valid_ravel_indices(key)
        if self.is_valid_arr_value(value, valid):
            value = value.reshape(-1)[valid]
        self._d[flat_key] = value
