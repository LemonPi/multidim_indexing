import torch
from multidim_indexing import torch_view as view


def test_index_2d():
    high = 50
    N = 15
    shape = (5, 10)
    data = torch.arange(0, high).reshape(shape)
    data_view = view.View(data)
    key_ravelled = torch.randint(high=high, size=(N,))
    key = view.unravel_index(key_ravelled, shape)

    query = data_view[key]
    # since the values are just a range, the ravelled key is the queried value
    assert torch.allclose(key_ravelled, query)


def test_index_multi_d():
    high = 500
    N = 15
    shape = (5, 10, 2, 5)
    data = torch.arange(0, high).reshape(shape)
    data_view = view.View(data)
    key_ravelled = torch.randint(high=high, size=(N,))
    key = view.unravel_index(key_ravelled, shape)

    query = data_view[key]
    # since the values are just a range, the ravelled key is the queried value
    assert torch.allclose(key_ravelled, query)


def test_set():
    high = 50
    N = 15
    shape = (5, 10)
    data = torch.arange(0, high).reshape(shape)
    data_view = view.View(data)
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


if __name__ == "__main__":
    test_index_2d()
    test_index_multi_d()
    test_set()
