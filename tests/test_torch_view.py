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
    key = view.unravel_index(key_ravelled, shape)

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
    key = view.unravel_index(key_ravelled, shape)

    query = data_view[key]
    # since the values are just a range, the ravelled key is the queried value
    assert torch.allclose(key_ravelled, query)


def test_value_2d():
    high = 50
    N = 15
    shape = (5, 10)
    data = torch.arange(0, high).reshape(shape)
    data_view = view.TorchMultidimView(data, value_ranges=[(0, 1), (0, 1)])
    key_ravelled = torch.randint(high=high, size=(N,))
    index_key = view.unravel_index(key_ravelled, shape)
    # convert to value between 0 and 1
    key = index_key / torch.tensor(shape)

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
    key = view.unravel_index(key_ravelled, shape)

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

    runs = 1000

    e_ours = []
    e_builtin = []
    e_builtin2 = []

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

        # since the values are just a range, the ravelled key is the queried value
        assert torch.allclose(key_ravelled, query)
        assert torch.allclose(key_ravelled, query_builtin)
        assert torch.allclose(key_ravelled, query_builtin2)
        e_ours.append(elapsed_ours)

    scale = 1000
    e_ours = np.array(e_ours) * scale
    e_builtin = np.array(e_builtin) * scale
    e_builtin2 = np.array(e_builtin2) * scale
    print(f"elapsed ours {np.mean(e_ours)} ({np.std(e_ours)}) builtin {np.mean(e_builtin)} ({np.std(e_builtin)}) "
          f"builtin2 {np.mean(e_builtin2)} ({np.std(e_builtin2)})")


if __name__ == "__main__":
    test_index_2d()
    test_index_multi_d()
    test_set()
    test_value_2d()
    test_performance()
