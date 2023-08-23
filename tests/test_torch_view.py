import time
import numpy as np
import torch
from multidim_indexing import torch_view as view


def test_index_2d():
    high = 50
    N = 15
    shape = (5, 10)
    data = torch.arange(0, high).reshape(shape)
    data_view = view.TorchMultidimView(data)
    key_ravelled = torch.randint(high=high, size=(N,))
    key = data_view.unravel_key(key_ravelled)

    query = data_view[key]
    # since the values are just a range, the ravelled key is the queried value
    assert torch.allclose(key_ravelled, query)

    # invalidate last key
    key[-1][0] = high
    query = data_view[key]
    # check that the last key is an invalid value
    assert query[-1] == data_view.invalid_value
    assert torch.allclose(key_ravelled[:-1], query[:-1])


def test_index_multi_d():
    high = 500
    N = 15
    shape = (5, 10, 2, 5)
    data = torch.arange(0, high).reshape(shape)
    data_view = view.TorchMultidimView(data)
    key_ravelled = torch.randint(high=high, size=(N,))
    key = data_view.unravel_key(key_ravelled)

    query = data_view[key]
    # since the values are just a range, the ravelled key is the queried value
    assert torch.allclose(key_ravelled, query)


def test_batch_key_value_query():
    high = 50
    B = 3
    N = 20
    shape = (5, 10)
    data = torch.arange(0, high).reshape(shape)
    data_view = view.TorchMultidimView(data.to(dtype=torch.float), value_ranges=[(0, 1), (0, 5)])

    key_ravelled = torch.randint(high=high, size=(N,))
    key = data_view.unravel_key(key_ravelled)
    batch_key = key.repeat(B, 1, 1)
    value_key = data_view.ensure_value_key(key)
    # value_key = torch.randn((N, len(shape)), dtype=torch.float)
    batch_value_key = value_key.repeat(B, 1, 1)
    assert not data_view.is_key_ravelled(key)
    assert not data_view.is_key_ravelled(batch_key)
    # test different ways of generating batch keys are equivalent
    batch_value_key2 = data_view.ensure_value_key(batch_key)
    assert torch.allclose(batch_value_key, batch_value_key2)
    batch_key2 = data_view.ensure_index_key(batch_value_key)
    assert torch.allclose(batch_key, batch_key2)

    query = data_view[value_key]
    # since the values are just a range, the ravelled key is the queried value
    assert torch.allclose(key_ravelled.to(dtype=torch.float), query)

    # test the usage of a batch key
    batch_query = data_view[batch_value_key]
    assert batch_query.shape == (B, N)
    for i in range(B):
        assert torch.allclose(batch_query[i], query)

    batch_value_key3 = torch.randn((B, N, len(shape)), dtype=torch.float)
    batch_query = data_view[batch_value_key3]
    for i in range(B):
        query = data_view[batch_value_key3[i]]
        assert torch.allclose(batch_query[i], query)


def test_value_2d():
    high = 50
    N = 15
    shape = (5, 10)
    data = torch.arange(0, high).reshape(shape)
    data_view = view.TorchMultidimView(data, value_ranges=[(0, 1), (0, 1)])
    key_ravelled = torch.randint(high=high, size=(N,))
    index_key = data_view.unravel_key(key_ravelled)
    # convert to value between 0 and 1
    key = index_key / (torch.tensor(shape) - 1)

    query = data_view[key]
    # since the values are just a range, the ravelled key is the queried value
    assert torch.allclose(key_ravelled, query)

    # invalidate last key
    key[-1][0] = high
    query = data_view[key]
    # check that the last key is an invalid value
    assert query[-1] == data_view.invalid_value
    assert torch.allclose(key_ravelled[:-1], query[:-1])


def test_set():
    high = 50
    N = 15
    shape = (5, 10)
    data = torch.arange(0, high).reshape(shape)
    data_view = view.TorchMultidimView(data)
    # having repeating indices results in undefined behavior
    key_ravelled = torch.randperm(high)[:N]
    key = data_view.unravel_key(key_ravelled)

    data_view[key] = -5
    # test that we changed the original data
    for i in range(N):
        assert int(data[tuple(key[i])]) == -5

    rand_val = torch.randint(low=-50, high=-5, size=(N,))
    data_view[key] = rand_val

    for i in range(N):
        assert data[tuple(key[i])] == rand_val[i]

    # invalidate last key
    key[-1][0] = high
    data_view[key] = rand_val[-1] + 1
    # check that we didn't change the location of the invalid key
    assert data.view(-1)[key_ravelled[-1]] == rand_val[-1]

    def index_sdf(cls, sdf, indices):
        # Assume SDF is B x 1 x (sdf_dim)
        # Assume indices is B x N x dx -- dx either 2 or 3
        B, N, dx = indices.shape
        nb = torch.arange(B).view(-1, 1)
        idxs = torch.chunk(indices, chunks=dx, dim=-1)
        idxs = [ix.squeeze(-1) for ix in idxs]
        if len(idxs) == 2:
            return sdf.squeeze(1)[nb, idxs[1], idxs[0]]

        if len(idxs) == 3:
            return sdf.squeeze(1)[nb, idxs[0], idxs[1], idxs[2]]


def test_performance():
    B = 256
    dx = 2
    N = 512
    shape = (B, 64, 64)
    high = torch.prod(torch.tensor(shape)).to(dtype=torch.long)
    data = torch.arange(0, high).reshape(shape)

    runs = 10

    e_ours = []
    e_builtin = []
    e_builtin2 = []
    e_shorthand = []

    for run in range(runs):
        key_ravelled = torch.randint(high=high, size=(B, N,))
        key = view.unravel_index(key_ravelled, shape)
        for i in range(B):
            key[i, :, 0] = i
        key_ravelled = view.ravel_multi_index(key, shape)

        key_builtin = key[:, :, 1:]
        start = time.time()
        nb = torch.arange(B).view(-1, 1)
        idxs = torch.chunk(key_builtin, chunks=dx, dim=-1)
        idxs = [ix.squeeze(-1) for ix in idxs]
        query_builtin = data[nb, idxs[0], idxs[1]]
        elapsed_builtin = time.time() - start
        e_builtin.append(elapsed_builtin)

        start = time.time()
        query_builtin2 = data[nb, key_builtin[..., 0], key_builtin[..., 1]]
        e_builtin2.append(time.time() - start)

        start = time.time()
        data_view = view.TorchMultidimView(data, check_safety=False)
        query = data_view[key]
        elapsed_ours = time.time() - start
        e_ours.append(elapsed_ours)

        start = time.time()
        # check that we can get the default behavior of batch indexing by default being applied to that batch
        # e.g. B x N x 2 indexing into B x X x Y
        data_view = view.TorchMultidimView(data, check_safety=False)
        query_shorthand = data_view[key_builtin]
        e_shorthand.append(time.time() - start)

        # since the values are just a range, the ravelled key is the queried value
        assert torch.allclose(key_ravelled, query)
        assert torch.allclose(key_ravelled, query_shorthand)
        assert torch.allclose(key_ravelled, query_builtin)
        assert torch.allclose(key_ravelled, query_builtin2)

    scale = 1000
    e_ours = np.array(e_ours) * scale
    e_shorthand = np.array(e_shorthand) * scale
    e_builtin = np.array(e_builtin) * scale
    e_builtin2 = np.array(e_builtin2) * scale
    print(f"elapsed ours {np.mean(e_ours)} ({np.std(e_ours)}) "
          f"ours shorthand {np.mean(e_shorthand)} ({np.std(e_shorthand)}) "
          f"builtin {np.mean(e_builtin)} ({np.std(e_builtin)}) "
          f"builtin2 {np.mean(e_builtin2)} ({np.std(e_builtin2)})")


def test_key_conversion():
    high = 10000
    N = 15
    shape = (5, 10, 20, 10)
    data = torch.arange(0, high).reshape(shape)
    data_view = view.TorchMultidimView(data.to(dtype=torch.float), value_ranges=[(0, 1), (0, 1), (-5, 5), (0, 10)])
    key_ravelled = torch.randint(high=high, size=(N,))
    key = data_view.unravel_key(key_ravelled)

    value_key = data_view.ensure_value_key(key)
    value_2_key = data_view.ensure_value_key(value_key)
    index_key = data_view.ensure_index_key(value_key)
    index_2_key = data_view.ensure_index_key(index_key)

    assert torch.allclose(key, index_key)
    assert torch.allclose(value_key, value_2_key)
    assert torch.allclose(index_key, index_2_key)

    value_key = torch.randn((N, len(shape)), dtype=torch.float)
    index_key = data_view.ensure_index_key(value_key)
    value_2_key = data_view.ensure_value_key(index_key)
    for i in range(len(shape)):
        assert torch.all(torch.abs(value_key[:, i] - value_2_key[:, i]) < data_view._resolution[i])


def test_ravel():
    high = 10000
    N = 15
    shape = (5, 10, 20, 10)
    data = torch.arange(0, high).reshape(shape)
    data_view = view.TorchMultidimView(data.to(dtype=torch.float))
    key_ravelled = torch.randint(high=high, size=(N,))
    key = data_view.unravel_key(key_ravelled)
    key_rereavelled = data_view.ravel_multi_index(key, shape)

    assert torch.allclose(key_ravelled, key_rereavelled)


def test_interpolation():
    from scipy.interpolate import interpn
    def value_func_3d(x, y, z):
        return 2 * x + 3 * y - z

    x = np.linspace(0, 4, 5)
    y = np.linspace(0, 5, 6)
    z = np.linspace(0, 6, 7)
    points = (x, y, z)
    values = value_func_3d(*np.meshgrid(*points, indexing='ij'))
    # randomly generate query points within the range
    N = 1000
    query = np.random.rand(N, 3) * np.array([4, 5, 6])
    gt = interpn(points, values, query, method='linear')

    m = view.TorchMultidimView(torch.tensor(values), [x, y, z], invalid_value=0, method='linear')
    v = m[torch.tensor(query)]

    assert torch.allclose(torch.tensor(gt), v)


if __name__ == "__main__":
    test_index_2d()
    test_index_multi_d()
    test_set()
    test_value_2d()
    test_performance()
    test_key_conversion()
    test_ravel()
    test_interpolation()
