# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn
from typing import List

import ocnn
from ocnn.octree import Octree


class OctreeGroupConv(torch.nn.Module):
  r''' Performs octree-based group convolution.

  Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    kernel_size (List(int)): The kernel shape, choose from :obj:`[3]`, :obj:`[2]`,
        :obj:`[3,3,3]`, :obj:`[3,1,1]`, :obj:`[1,3,1]`, :obj:`[1,1,3]`,
        :obj:`[2,2,2]`, :obj:`[3,3,1]`, :obj:`[1,3,3]`, and :obj:`[3,1,3]`.
    stride (int): The stride of the convolution (:obj:`1` or :obj:`2`).
    nempty (bool): If True, only performs the convolution on non-empty
        octree nodes.
    use_bias (bool): If True, add a bias term to the convolution.
    group (int): The number of groups.

  .. note::
    Perform octree-based group convolution with a for-loop. The performance is
    not optimal. Use this module only when the group number is small, otherwise
    it may be slow.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False, use_bias: bool = False,
               group: int = 1):
    super().__init__()

    self.group = group
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.in_channels_per_group = in_channels // group
    self.out_channels_per_group = out_channels // group
    assert in_channels % group == 0 and out_channels % group == 0

    self.convs = torch.nn.ModuleList([ocnn.nn.OctreeConv(
        self.in_channels_per_group, self.out_channels_per_group,
        kernel_size, stride, nempty, use_bias=use_bias)
        for _ in range(group)])

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''' Defines the octree-based group convolution.

    Args:
      data (torch.Tensor): The input data.
      octree (Octree): The corresponding octree.
      depth (int): The depth of current octree.
    '''

    channels = data.shape[1]
    assert channels == self.in_channels

    outs = [None] * self.group
    slices = torch.split(data, self.in_channels_per_group, dim=1)
    for i in range(self.group):
      outs[i] = self.convs[i](slices[i], octree, depth)
    out = torch.cat(outs, dim=1)
    return out

  def extra_repr(self) -> str:
    r''' Sets the extra representation of the module.
    '''

    return ('in_channels={}, out_channels={}, group={}').format(
             self.in_channels, self.out_channels, self.group)  # noqa
