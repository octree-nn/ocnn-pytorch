# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn
from typing import Optional

from ocnn.octree import Octree
from ocnn.utils import scatter_add


OctreeBatchNorm = torch.nn.BatchNorm1d


class OctreeGroupNorm(torch.nn.Module):
  r''' An group normalization layer for the octree.
  '''

  def __init__(self, in_channels: int, group: int, nempty: bool = False,
               min_group_channels: int = 4):
    super().__init__()
    self.eps = 1e-5
    self.nempty = nempty
    self.group = group
    self.in_channels = in_channels
    self.min_group_channels = min_group_channels
    if self.min_group_channels * self.group > in_channels:
      self.group = in_channels // self.min_group_channels

    assert in_channels % self.group == 0
    self.channels_per_group = in_channels // self.group

    self.weights = torch.nn.Parameter(torch.Tensor(1, in_channels))
    self.bias = torch.nn.Parameter(torch.Tensor(1, in_channels))
    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.ones_(self.weights)
    torch.nn.init.zeros_(self.bias)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    batch_size = octree.batch_size
    batch_id = octree.batch_id(depth, self.nempty)
    ones = data.new_ones([data.shape[0], 1])
    count = scatter_add(ones, batch_id, dim=0, dim_size=batch_size)
    count = count * self.channels_per_group  # element number in each group
    inv_count = 1.0 / (count + self.eps)  # there might be 0 element sometimes

    mean = scatter_add(data, batch_id, dim=0, dim_size=batch_size) * inv_count
    mean = self._adjust_for_group(mean)
    out = data - mean.index_select(0, batch_id)

    var = scatter_add(out**2, batch_id, dim=0, dim_size=batch_size) * inv_count
    var = self._adjust_for_group(var)
    inv_std = 1.0 / (var + self.eps).sqrt()
    out = out * inv_std.index_select(0, batch_id)

    out = out * self.weights + self.bias
    return out

  def _adjust_for_group(self, tensor: torch.Tensor):
    r''' Adjust the tensor for the group.
    '''

    if self.channels_per_group > 1:
      tensor = (tensor.reshape(-1, self.group, self.channels_per_group)
                      .sum(-1, keepdim=True)
                      .repeat(1, 1, self.channels_per_group)
                      .reshape(-1, self.in_channels))
    return tensor

  def extra_repr(self) -> str:
    return ('in_channels={}, group={}, nempty={}').format(
            self.in_channels, self.group, self.nempty)  # noqa


class OctreeInstanceNorm(OctreeGroupNorm):
  r''' An instance normalization layer for the octree.
  '''

  def __init__(self, in_channels: int, nempty: bool = False):
    super().__init__(in_channels=in_channels, group=in_channels, nempty=nempty)

  def extra_repr(self) -> str:
    return ('in_channels={}, nempty={}').format(self.in_channels, self.nempty)


class OctreeNorm(torch.nn.Module):
  r''' A normalization layer for the octree. It encapsulates octree-based batch,
  group and instance normalization.
  '''

  def __init__(self, in_channels: int, norm_type: str = 'batch_norm',
               group: int = 32, min_group_channels: int = 4):
    super().__init__()
    self.in_channels = in_channels
    self.norm_type = norm_type
    self.group = group
    self.min_group_channels = min_group_channels

    if self.norm_type == 'batch_norm':
      self.norm = torch.nn.BatchNorm1d(in_channels)
    elif self.norm_type == 'group_norm':
      self.norm = OctreeGroupNorm(in_channels, group, min_group_channels)
    elif self.norm_type == 'instance_norm':
      self.norm = OctreeInstanceNorm(in_channels)
    else:
      raise ValueError

  def forward(self, x: torch.Tensor, octree: Optional[Octree] = None,
              depth: Optional[int] = None):
    if self.norm_type == 'batch_norm':
      output = self.norm(x)
    elif (self.norm_type == 'group_norm' or
          self.norm_type == 'instance_norm'):
      output = self.norm(x, octree, depth)
    else:
      raise ValueError
    return output
