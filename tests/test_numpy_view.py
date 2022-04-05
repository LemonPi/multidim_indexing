import numpy as np
from multidim_indexing import numpy_view as view


def test_index_2d():
    high = 50
    N = 15
    shape = (5, 10)
    data = np.arange(0, high).reshape(shape)
    data_view = view.NumpyView(data)
    key_ravelled = np.random.randint(0, high=high, size=(N,))
    key = np.stack(np.unravel_index(key_ravelled, shape)).T

    query = data_view[key]
    # since the values are just a range, the ravelled key is the queried value
    assert np.allclose(key_ravelled, query)

    # invalidate last key
    key[-1][0] = high
    query = data_view[key]
    # check that the last key is an invalid value
    assert query[-1] == data_view.invalid_value
    assert np.allclose(key_ravelled[:-1], query[:-1])


def test_index_multi_d():
    high = 500
    N = 15
    shape = (5, 10, 2, 5)
    data = np.arange(0, high).reshape(shape)
    data_view = view.NumpyView(data)
    key_ravelled = np.random.randint(0, high=high, size=(N,))
    key = np.stack(np.unravel_index(key_ravelled, shape)).T

    query = data_view[key]
    # since the values are just a range, the ravelled key is the queried value
    assert np.allclose(key_ravelled, query)


def test_value_2d():
    high = 50
    N = 15
    shape = (5, 10)
    data = np.arange(0, high).reshape(shape)
    data_view = view.NumpyView(data, value_ranges=[(0, 1), (0, 1)])
    key_ravelled = np.random.randint(0, high=high, size=(N,))
    index_key = np.stack(np.unravel_index(key_ravelled, shape)).T
    # convert to value between 0 and 1
    key = index_key / np.array(shape)

    query = data_view[key]
    # since the values are just a range, the ravelled key is the queried value
    assert np.allclose(key_ravelled, query)

    # invalidate last key
    key[-1][0] = high
    query = data_view[key]
    # check that the last key is an invalid value
    assert query[-1] == data_view.invalid_value
    assert np.allclose(key_ravelled[:-1], query[:-1])


def test_set():
    high = 50
    N = 15
    shape = (5, 10)
    data = np.arange(0, high).reshape(shape)
    data_view = view.NumpyView(data)
    # having repeating indices results in undefined behavior
    key_ravelled = np.random.permutation(high)[:N]
    key = np.stack(np.unravel_index(key_ravelled, shape)).T

    data_view[key] = -5
    # test that we changed the original data
    for i in range(N):
        assert int(data[tuple(key[i])]) == -5

    rand_val = np.random.randint(-50, high=-5, size=(N,))
    data_view[key] = rand_val

    for i in range(N):
        assert data[tuple(key[i])] == rand_val[i]

    # invalidate last key
    key[-1][0] = high
    data_view[key] = rand_val[-1] + 1
    # check that we didn't change the location of the invalid key
    assert data.reshape(-1)[key_ravelled[-1]] == rand_val[-1]


if __name__ == "__main__":
    test_index_2d()
    test_index_multi_d()
    test_set()
    test_value_2d()
