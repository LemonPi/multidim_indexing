import time
import numpy as np
from multidim_indexing import numpy_view as view


def test_index_2d():
    high = 50
    N = 15
    shape = (5, 10)
    data = np.arange(0, high).reshape(shape)
    data_view = view.NumpyMultidimView(data)
    key_ravelled = np.random.randint(0, high=high, size=(N,))
    key = data_view.unravel_key(key_ravelled)

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
    data_view = view.NumpyMultidimView(data)
    key_ravelled = np.random.randint(0, high=high, size=(N,))
    key = data_view.unravel_key(key_ravelled)

    query = data_view[key]
    # since the values are just a range, the ravelled key is the queried value
    assert np.allclose(key_ravelled, query)


def test_batch_key_value_query():
    high = 50
    B = 3
    N = 20
    shape = (5, 10)
    data = np.arange(0, high).reshape(shape)
    data_view = view.NumpyMultidimView(data.astype(dtype=float), value_ranges=[(0, 1), (0, 5)])

    key_ravelled = np.random.randint(0, high=high, size=(N,))
    key = data_view.unravel_key(key_ravelled)
    batch_key = np.tile(key, (B, 1, 1))
    value_key = data_view.ensure_value_key(key)
    # value_key = torch.randn((N, len(shape)), dtype=torch.float)
    batch_value_key = np.tile(value_key, (B, 1, 1))
    assert not data_view.is_key_ravelled(key)
    assert not data_view.is_key_ravelled(batch_key)
    # test different ways of generating batch keys are equivalent
    batch_value_key2 = data_view.ensure_value_key(batch_key)
    assert np.allclose(batch_value_key, batch_value_key2)
    batch_key2 = data_view.ensure_index_key(batch_value_key)
    assert np.allclose(batch_key, batch_key2)

    query = data_view[value_key]
    # since the values are just a range, the ravelled key is the queried value
    assert np.allclose(key_ravelled.astype(dtype=float), query)

    # test the usage of a batch key
    batch_query = data_view[batch_value_key]
    assert batch_query.shape == (B, N)
    for i in range(B):
        assert np.allclose(batch_query[i], query)

    batch_value_key3 = np.random.randn(B, N, len(shape))
    batch_query = data_view[batch_value_key3]
    for i in range(B):
        query = data_view[batch_value_key3[i]]
        assert np.allclose(batch_query[i], query)


def test_value_2d():
    high = 50
    N = 15
    shape = (5, 10)
    data = np.arange(0, high).reshape(shape)
    data_view = view.NumpyMultidimView(data, value_ranges=[(0, 1), (0, 1)])
    key_ravelled = np.random.randint(0, high=high, size=(N,))
    index_key = data_view.unravel_key(key_ravelled)
    # convert to value between 0 and 1
    key = index_key / (np.array(shape) - 1)

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
    data_view = view.NumpyMultidimView(data)
    # having repeating indices results in undefined behavior
    key_ravelled = np.random.permutation(high)[:N]
    key = data_view.unravel_key(key_ravelled)

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


def test_performance():
    B = 256
    dx = 2
    N = 512
    shape = (B, 64, 64)
    high = np.prod(np.array(shape)).astype(dtype=int)
    data = np.arange(0, high).reshape(shape)

    runs = 10

    e_ours = []
    e_builtin = []
    e_builtin2 = []
    e_shorthand = []

    for run in range(runs):
        key_ravelled = np.random.randint(0, high=high, size=(B, N,))
        key = np.stack(np.unravel_index(key_ravelled, shape), axis=0)
        for i in range(B):
            key[0, i, :] = i
        key_ravelled = np.ravel_multi_index(key, shape)
        key = key.transpose((1, 2, 0))

        key_builtin = key[:, :, 1:]
        start = time.time()
        nb = np.arange(B).reshape(-1, 1)
        idxs = np.array_split(key_builtin, key_builtin.shape[-1], axis=-1)
        idxs = [ix.squeeze(-1) for ix in idxs]
        query_builtin = data[nb, idxs[0], idxs[1]]
        elapsed_builtin = time.time() - start
        e_builtin.append(elapsed_builtin)

        start = time.time()
        query_builtin2 = data[nb, key_builtin[:, :, 0], key_builtin[:, :, 1]]
        e_builtin2.append(time.time() - start)

        start = time.time()
        data_view = view.NumpyMultidimView(data, check_safety=False)
        query = data_view[key]
        elapsed_ours = time.time() - start
        e_ours.append(elapsed_ours)

        start = time.time()
        # check that we can get the default behavior of batch indexing by default being applied to that batch
        # e.g. B x N x 2 indexing into B x X x Y
        data_view = view.NumpyMultidimView(data, check_safety=False)
        query_shorthand = data_view[key_builtin]
        e_shorthand.append(time.time() - start)

        # since the values are just a range, the ravelled key is the queried value
        assert np.allclose(key_ravelled, query)
        assert np.allclose(key_ravelled, query_shorthand)
        assert np.allclose(key_ravelled, query_builtin)
        assert np.allclose(key_ravelled, query_builtin2)

    scale = 1000
    e_ours = np.array(e_ours) * scale
    e_shorthand = np.array(e_shorthand) * scale
    e_builtin = np.array(e_builtin) * scale
    e_builtin2 = np.array(e_builtin2) * scale
    print(f"elapsed ours {np.mean(e_ours)} ({np.std(e_ours)}) "
          f"ours shorthand {np.mean(e_shorthand)} ({np.std(e_shorthand)}) "
          f"builtin {np.mean(e_builtin)} ({np.std(e_builtin)}) "
          f"builtin2 {np.mean(e_builtin2)} ({np.std(e_builtin2)})")


if __name__ == "__main__":
    test_index_2d()
    test_index_multi_d()
    test_set()
    test_value_2d()
    test_performance()
