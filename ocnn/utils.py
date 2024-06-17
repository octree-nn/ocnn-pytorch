# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import math
import torch
from typing import Optional
from packaging import version

import ocnn


__all__ = ['trunc_div', 'meshgrid', 'cumsum', 'scatter_add', 'xavier_uniform_',
           'resize_with_last_val', 'list2str', 'build_example_octree']
classes = __all__


def trunc_div(input, other):
  r''' Wraps :func:`torch.div` for compatibility. It rounds the results of the
  division towards zero and is equivalent to C-style integer  division.
  '''

  larger_than_171 = version.parse(torch.__version__) > version.parse('1.7.1')

  if larger_than_171:
    return torch.div(input, other, rounding_mode='trunc')
  else:
    return torch.floor_divide(input, other)


def meshgrid(*tensors, indexing: Optional[str] = None):
  r''' Wraps :func:`torch.meshgrid` for compatibility.
  '''

  larger_than_191 = version.parse(torch.__version__) > version.parse('1.9.1')

  if larger_than_191:
    return torch.meshgrid(*tensors, indexing=indexing)
  else:
    return torch.meshgrid(*tensors)


def range_grid(min: int, max: int, device: torch.device = 'cpu'):
  r''' Builds a 3D mesh grid in :obj:`[min, max]` (:attr:`max` included).

  Args:
    min (int): The minimum value of the grid.
    max (int): The maximum value of the grid.
    device (torch.device, optional): The device to place the grid on.

  Returns:
    torch.Tensor: A 3D mesh grid tensor of shape (N, 3), where N is the total
                  number of grid points.

  Example:
    >>> grid = range_grid(0, 1)
    >>> print(grid)
    tensor([[0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]])
  '''

  rng = torch.arange(min, max+1, dtype=torch.long, device=device)
  grid = meshgrid(rng, rng, rng, indexing='ij')
  grid = torch.stack(grid, dim=-1).view(-1, 3)
  return grid


def cumsum(data: torch.Tensor, dim: int, exclusive: bool = False):
  r''' Extends :func:`torch.cumsum` with the input argument :attr:`exclusive`.

  Args:
    data (torch.Tensor): The input data.
    dim (int): The dimension to do the operation over.
    exclusive (bool): If false, the behavior is the same as :func:`torch.cumsum`;
        if true, returns the cumulative sum exclusively. Note that if ture,
        the shape of output tensor is larger by 1 than :attr:`data` in the
        dimension where the computation occurs.
  '''

  out = torch.cumsum(data, dim)

  if exclusive:
    size = list(data.size())
    size[dim] = 1
    zeros = out.new_zeros(size)
    out = torch.cat([zeros, out], dim)
  return out


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
  r''' Broadcast :attr:`src` according to :attr:`other`, originally from the
  library `pytorch_scatter`.
  '''

  if dim < 0:
    dim = other.dim() + dim

  if src.dim() == 1:
    for _ in range(0, dim):
      src = src.unsqueeze(0)
  for _ in range(src.dim(), other.dim()):
    src = src.unsqueeze(-1)

  src = src.expand_as(other)
  return src


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,) -> torch.Tensor:
  r''' Reduces all values from the :attr:`src` tensor into :attr:`out` at the
  indices specified in the :attr:`index` tensor along a given axis :attr:`dim`.
  This is just a wrapper of :func:`torch.scatter` in a boardcasting fashion.

  Args:
    src (torch.Tensor): The source tensor.
    index (torch.Tensor): The indices of elements to scatter.
    dim (torch.Tensor): The axis along which to index, (default: :obj:`-1`).
    out (torch.Tensor or None): The destination tensor.
    dim_size (int or None): If :attr:`out` is not given, automatically create
        output with size :attr:`dim_size` at dimension :attr:`dim`. If
        :attr:`dim_size` is not given, a minimal sized output tensor according
        to :obj:`index.max() + 1` is returned.
    '''

  index = broadcast(index, src, dim)

  if out is None:
    size = list(src.size())
    if dim_size is not None:
      size[dim] = dim_size
    elif index.numel() == 0:
      size[dim] = 0
    else:
      size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)

  return out.scatter_add_(dim, index, src)


def xavier_uniform_(weights: torch.Tensor):
  r''' Initialize convolution weights with the same method as
  :obj:`torch.nn.init.xavier_uniform_`.

  :obj:`torch.nn.init.xavier_uniform_` initialize a tensor with shape
  :obj:`(out_c, in_c, kdim)`, which can not be used in :class:`ocnn.nn.OctreeConv`
  since the the shape of :attr:`OctreeConv.weights` is :obj:`(kdim, in_c,
  out_c)`.
  '''

  shape = weights.shape     # (kernel_dim, in_conv, out_conv)
  fan_in = shape[0] * shape[1]
  fan_out = shape[0] * shape[2]
  std = math.sqrt(2.0 / float(fan_in + fan_out))
  a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

  torch.nn.init.uniform_(weights, -a, a)


def resize_with_last_val(list_in: list, num: int = 3):
  r''' Resizes the number of elements of :attr:`list_in` to :attr:`num` with
  the last element of :attr:`list_in` if its number of elements is smaller
  than :attr:`num`.
  '''

  assert (type(list_in) is list and len(list_in) < num + 1)
  for i in range(len(list_in), num):
    list_in.append(list_in[-1])
  return list_in


def list2str(list_in: list):
  r''' Returns a string representation of :attr:`list_in`.
  '''

  out = [str(x) for x in list_in]
  return ''.join(out)


def build_example_octree(depth: int = 5, full_depth: int = 2, pt_num: int = 3):
  r''' Builds an example octree on CPU from at most 3 points.
  '''
  # initialize the point cloud
  points = torch.Tensor([[-1, -1, -1], [0, 0, -1], [0.0625, 0.0625, -1]])
  normals = torch.Tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
  features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
  labels = torch.Tensor([[0], [2], [2]])

  assert pt_num <= 3 and pt_num > 0
  point_cloud = ocnn.octree.Points(
      points[:pt_num], normals[:pt_num], features[:pt_num], labels[:pt_num])

  # build octree
  octree = ocnn.octree.Octree(depth, full_depth)
  octree.build_octree(point_cloud)
  return octree
