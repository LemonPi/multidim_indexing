import time
import numpy as np
import torch
from multidim_indexing import torch_view as view
from pytorch_seed import seed

d = "cuda" if torch.cuda.is_available() else "cpu"


# d = "cpu"

def test_query_performance():
    seed(0)
    data_size = 100
    query_size = 20000

    data = torch.randn(data_size, data_size, data_size, device=d)
    data_view = view.TorchMultidimView(data, value_ranges=[(0, 1), (0, 5), (0, 10)])

    times = []
    res = None
    for iter in range(10):
        query = torch.randn((query_size, 3), device=d)
        start = time.time()
        this_res = data_view[query]
        if res is None:
            res = this_res
        else:
            res += this_res
        end = time.time()
        times.append(end - start)

    print(f"{res.mean()}")
    print(f"Average query time: {np.median(times)}")
    print(f"All query times: {times}")


def test_linear_query_performance():
    seed(0)
    data_size = 100
    query_size = 200000

    data = torch.randn(data_size, data_size, data_size, device=d)
    data_view = view.TorchMultidimView(data, value_ranges=[(0, 1), (0, 5), (0, 10)], method="linear")

    times = []
    res = None
    for iter in range(10):
        query = torch.randn((query_size, 3), device=d)
        start = time.time()
        this_res = data_view[query]
        if res is None:
            res = this_res
        else:
            res += this_res
        end = time.time()
        times.append(end - start)

    print(f"{res.mean()}")
    print(f"Average query time: {np.median(times)}")
    print(f"All query times: {times}")


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

    assert torch.allclose(torch.tensor(gt), v, atol=1e-6)


if __name__ == "__main__":
    test_interpolation()
    test_query_performance()
    test_linear_query_performance()
