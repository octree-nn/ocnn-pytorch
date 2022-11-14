# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn
from typing import List

from ocnn.octree import Octree
from ocnn.utils import meshgrid, scatter_add, resize_with_last_val, list2str
from . import octree_pad, octree_depad


def octree_max_pool(data: torch.Tensor, octree: Octree, depth: int,
                    nempty: bool = False, return_indices: bool = False):
  r''' Performs octree max pooling with kernel size 2 and stride 2.

  Args:
    data (torch.Tensor): The input tensor.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current octree. After pooling, the corresponding
        depth decreased by 1.
    nempty (bool): If True, :attr:`data` contains only features of non-empty
        octree nodes.
    return_indices (bool): If True, returns the indices, which can be used in
        :func:`octree_max_unpool`.
  '''

  if nempty:
    data = octree_pad(data, octree, depth, float('-inf'))
  data = data.view(-1, 8, data.shape[1])
  out, indices = data.max(dim=1)
  if not nempty:
    out = octree_pad(out, octree, depth-1)
  return (out, indices) if return_indices else out


def octree_max_unpool(data: torch.Tensor, indices: torch.Tensor, octree: Octree,
                      depth: int, nempty: bool = False):
  r''' Performs octree max unpooling.

  Args:
    data (torch.Tensor): The input tensor.
    indices (torch.Tensor): The indices returned by :func:`octree_max_pool`. The
        depth of :attr:`indices` is larger by 1 than :attr:`data`.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current data. After unpooling, the corresponding
        depth increases by 1.
  '''

  if not nempty:
    data = octree_depad(data, octree, depth)
  num, channel = data.shape
  out = torch.zeros(num, 8, channel, dtype=data.dtype, device=data.device)
  i = torch.arange(num, dtype=indices.dtype, device=indices.device)
  k = torch.arange(channel, dtype=indices.dtype, device=indices.device)
  i, k = meshgrid(i, k, indexing='ij')
  out[i, indices, k] = data
  out = out.view(-1, channel)
  if nempty:
    out = octree_depad(out, octree, depth+1)
  return out


def octree_avg_pool(data: torch.Tensor, octree: Octree, depth: int,
                    kernel: str, stride: int = 2, nempty: bool = False):
  r''' Performs octree average pooling.

  Args:
    data (torch.Tensor): The input tensor.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current octree.
    kernel (str): The kernel size, like '333', '222'.
    stride (int): The stride of the pooling.
    nempty (bool): If True, :attr:`data` contains only features of non-empty
        octree nodes.
  '''

  neigh = octree.get_neigh(depth, kernel, stride, nempty)

  N1 = data.shape[0]
  N2 = neigh.shape[0]
  K = neigh.shape[1]

  mask = neigh >= 0
  val = 1.0 / (torch.sum(mask, dim=1) + 1e-8)
  mask = mask.view(-1)
  val = val.unsqueeze(1).repeat(1, K).reshape(-1)
  val = val[mask]

  row = torch.arange(N2, device=neigh.device)
  row = row.unsqueeze(1).repeat(1, K).view(-1)
  col = neigh.view(-1)
  indices = torch.stack([row[mask], col[mask]], dim=0).long()

  mat = torch.sparse_coo_tensor(indices, val, [N2, N1], device=data.device)
  out = torch.sparse.mm(mat, data)
  return out


def octree_global_pool(data: torch.Tensor, octree: Octree, depth: int,
                       nempty: bool = False):
  r''' Performs octree global average pooling.

  Args:
    data (torch.Tensor): The input tensor.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current octree.
    nempty (bool): If True, :attr:`data` contains only features of non-empty
        octree nodes.
  '''

  batch_size = octree.batch_size
  batch_id = octree.batch_id(depth, nempty)
  ones = data.new_ones(data.shape[0], 1)
  count = scatter_add(ones, batch_id, dim=0, dim_size=batch_size)
  count[count < 1] = 1  # there might be 0 element in some shapes

  out = scatter_add(data, batch_id, dim=0, dim_size=batch_size)
  out = out / count
  return out


class OctreePoolBase(torch.nn.Module):
  r''' The base class for octree-based pooling.
  '''

  def __init__(self, kernel_size: List[int], stride: int, nempty: bool = False):
    super().__init__()
    self.kernel_size = resize_with_last_val(kernel_size)
    self.kernel = list2str(self.kernel_size)
    self.stride = stride
    self.nempty = nempty

  def extra_repr(self) -> str:
    return ('kernel_size={}, stride={}, nempty={}').format(
            self.kernel_size, self.stride, self.nempty)  # noqa


class OctreeMaxPool(OctreePoolBase):
  r''' Performs octree max pooling.

  Please refer to :func:`octree_max_pool` for details.
  '''

  def __init__(self, nempty: bool = False, return_indices: bool = False):
    super().__init__(kernel_size=[2], stride=2, nempty=nempty)
    self.return_indices = return_indices

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    return octree_max_pool(data, octree, depth, self.nempty, self.return_indices)


class OctreeMaxUnpool(OctreePoolBase):
  r''' Performs octree max unpooling.

  Please refer to :func:`octree_max_unpool` for details.
  '''

  def forward(self, data: torch.Tensor, indices: torch.Tensor, octree: Octree,
              depth: int):
    r''''''

    return octree_max_unpool(data, indices, octree, depth, self.nempty)


class OctreeGlobalPool(OctreePoolBase):
  r''' Performs octree global pooling.

  Please refer to :func:`octree_global_pool` for details.
  '''

  def __init__(self, nempty: bool = False):
    super().__init__(kernel_size=[-1], stride=-1, nempty=nempty)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    return octree_global_pool(data, octree, depth, self.nempty)


class OctreeAvgPool(OctreePoolBase):
  r''' Performs octree average pooling.

  Please refer to :func:`octree_avg_pool` for details.
  '''

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    return octree_avg_pool(
        data, octree, depth, self.kernel, self.stride, self.nempty)
