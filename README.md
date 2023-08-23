This repository documents the syntax for multidimensional indexing for Pytorch and Numpy, and offers classes that
encapsulates the process and provides additional features on top for data that represents a coordinate grid.
You can follow along the code blocks here with the included Jupyter notebook.

## Multidimensional Indexing

Suppose we have a multidimensional tensor, which could be a cached voxel grid, or a batch of images
(the values are ordered to make clear how the indexing works):

```python
import torch

B = 256  # batch size (optional)
shape = (B, 64, 64)
high = torch.prod(torch.tensor(shape)).to(dtype=torch.long)
data = torch.arange(0, high).reshape(shape)
```

A key operation on this tensor is to index it for querying and assignment. It is straightforward to index a single
value, and particular groupings of dimensions:

```python
# index a single element
print(data[124, 5, 52])

# index all dimensions given the first is index 0 (the following are equivalent)
print(data[0])
print(data[0, :, :])
print(data[0, ...])  # pytorch only syntax

# index all dimensions given the last is index 5 (the following are equivalent)
print(data[..., 5])
print(data[:, :, 5])
```

It is also straightforward to batch index along a single dimension:

```python
idx = [4, 8, 15, 16, 23, 42]

# index all dimensions given the first follows idx
print(data[idx].shape)  # (len(idx), 64, 64)
print(data[idx, ...].shape)
print(data[idx, :, :].shape)

# index all dimensions given the second follows idx
print(data[:, idx].shape)
print(data[:, idx, :].shape)
```

It is also reasonable to batch index along multiple dimensions. Note that it does not make sense for `idx` and `idx2` to
have different lengths since that would lead to combinations where one is missing a value.

```python
idx = [4, 8, 15, 16, 23, 42]
idx2 = [5, 2, 7, 1, 32, 4]

# index the last dimension when the first two are (4,5), (8,2), (15,7), (16,1), (23,32), and (42,4)
print(data[idx, idx2].shape)  # (len(idx), 64)
```

It is also common to have a list of entries by their indices that we'd like to batch query.

```python
# indices of 5 entries
idx3 = [[0, 5, 3],
        [2, 7, 5],
        [100, 23, 45],
        [3, 6, 4],
        [4, 2, 1]]
```

Directly indexing the tensor with a multidimensional index does not do what you want:

```python
print(data[idx3])  # results in an error
```

Instead, **split up the indices by their dimension** either manually, or with `torch.unbind`

```python
# easier to convert it to something that allows column indexing first
idx4 = torch.tensor(idx3)
print(data[idx4[:, 0], idx4[:, 1], idx4[:, 2]])  # returns the 5 entries as desired
print(data[torch.unbind(idx4, -1)])              # can also use unbind
```

## How can it be improved?

Most importantly, it may not be clear why simply doing `data[idx3]` does not work, and what the correct syntax is. So
reading up to here should resolve most questions about indexing with a batch of indices on a multidimensional tensor.
This library provides `MultidimView` variants (torch and numpy) that provide a view for these tensors with features
specialized to multidimensional tensor that represent coordinate gridded values:

- direct indexing so `data[idx3]` does what you want
- optional indexing on values if you specify value ranges
    - value resolution implicitly defined by size of source and value range
- optional safety checking for out of bound values or indices
    - provide default value for out of bound queries instead of throwing an exception

## Installation
numpy only
```shell
pip install multidim-indexing[numpy]
```
pytorch only
```shell
pip install multidim-indexing[torch]
```
all
```shell
pip install multidim-indexing[all]
```
## Usage

Continuing with `data` and the indices described before,

```python
from multidim_indexing import torch_view as view

# for numpy, import numpy_view and use NumpyMultidimView

# simple wrapper with bounds checking
data_multi = view.TorchMultidimView(data)
# another view into the data, treating it as a batch of 2 dimensional grid data with X in [-5, 5] and Y in [0, 10]
# can specify value to assign a query if it's out of bounds (defaults to -1)
# note that the invalid value needs to be of the same type as the source, so we can't for example use float('inf') here
data_batch = view.TorchMultidimView(data, value_ranges=[[0, B], [-5, 5], [0, 10]], invalid_value=-1)
# another view into the data, treating it as a 3D grid data with X in [-2.5, 5], Y in [0, 4], and Z in [0, 10]
data_3d = view.TorchMultidimView(data, value_ranges=[[-2.5, 5], [0, 4], [0, 10]])
```
By default, the nearest grid value is returned. You can instead use linear interpolation like scipy's interpn by setting
`method='linear'` in the constructor.
```python
data_3d = view.TorchMultidimView(data, value_ranges=[[-2.5, 5], [0, 4], [0, 10]], method='linear')
```

We can then use them like:

```python
# convert index to the corresponding type (pytorch vs numpy)
key = torch.tensor(idx3, dtype=torch.long)
print(data_multi[key])  # returns the 5 entries as desired
```

```python
# query the other views using grid values
# first, let's try keying the same 2D values across all batches
value_key_per_batch = torch.tensor([[-3.5, 0.2],
                                    [-4, 0.1],
                                    [-7, 0.5],  # this is out of bounds
                                    [3, 2]])
# number of entries to query
N = value_key_per_batch.shape[0]
print(torch.arange(B, dtype=value_key_per_batch.dtype).reshape(B, 1, 1).repeat(1, N, 1).shape)
# make the indices for all batches
value_key_batch = torch.cat(
    (torch.arange(B, dtype=value_key_per_batch.dtype).reshape(B, 1, 1).repeat(1, N, 1),
     value_key_per_batch.repeat(B, 1, 1)), dim=-1)
# keys can have an additional batch indices at the front
print(value_key_batch.shape)  # (B, N, 3)
# these 2 should be the same apart from the first batch index
print(value_key_batch[0:N])
print(value_key_batch[12*N:13*N])

# should see some -1 to indicate invalid value
print(data_batch[value_key_batch]) 

# also there is a shorthand for directly using the per batch indices
print(data_batch[value_key_per_batch.repeat(B,1,1)]) # should be the same as above
```

```python
value_key_3d = torch.tensor([[-2.5, 0., 0.],  # right on the boundary of validity
                             [-2.51, 0.5, 0],  # out of bounds
                             [5, 4, 10]  # right on the boundary
                             ]
                            )
print(data_3d[value_key_3d])  # (0, -1 for invalid, high - 1)
print(torch.prod(torch.tensor(data.shape)) - 1)
print(high - 1)
```

The indexing naturally allows setting in addition to querying. Out of bound indices will be
ignored.
