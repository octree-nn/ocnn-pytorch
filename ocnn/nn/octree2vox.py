# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch

from ..octree import Octree, key2xyz
from .octree_pad import octree_depad


def octree2voxel(data: torch.Tensor, octree: Octree, depth: int,
                 nempty: bool = False):
  r''' Converts the input feature to full-voxel-based representation.

  Args:
    data (torch.Tensor): The input feature.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current octree.
    nempty (bool): If True, :attr:`data` only contains the features of non-empty
        octree nodes.
  '''

  key = octree.keys[depth]
  if nempty:
    key = octree_depad(key, octree, depth)
  x, y, z, b = key2xyz(key, depth)

  num = 1 << depth
  channel = data.shape[1]
  batch_size = octree.batch_size
  size = (batch_size, num, num, num, channel)
  vox = torch.zeros(size, dtype=data.dtype, device=data.device)
  vox[b, x, y, z] = data
  return vox


class Octree2Voxel(torch.nn.Module):
  r''' Converts the input feature to full-voxel-based representation

  Please refer to :func:`octree2voxel` for details.
  '''

  def __init__(self, nempty: bool = False):
    super().__init__()
    self.nempty = nempty

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    return octree2voxel(data, octree, depth, self.nempty)

  def extra_repr(self) -> str:
    return 'nempty={}'.format(self.nempty)
