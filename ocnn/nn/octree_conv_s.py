# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn

from ocnn.octree import Octree
from ocnn.utils import xavier_uniform_
from ocnn.nn.octree_pad import octree_pad


class OctreeConvK2(torch.nn.Module):
  r''' Performs octree convolution with kernel size 2 and stride 2.

  Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    nempty (bool): If True, only performs the convolution on non-empty octree
        nodes; otherwise, performs the convolution on all octree nodes.
    use_bias (bool): If True, add a bias term to the convolution.

  '''

  def __init__(self, in_channels: int, out_channels: int,
               nempty: bool = False, use_bias: bool = False, **kwargs):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.nempty = nempty
    self.use_bias = use_bias

    weights_shape = (8, in_channels, out_channels)
    self.weights = torch.nn.Parameter(torch.Tensor(*weights_shape))
    self.bias = (torch.nn.Parameter(torch.Tensor(self.out_channels))
                 if use_bias else None)
    self.reset_parameters()

  def reset_parameters(self):
    xavier_uniform_(self.weights)
    if self.use_bias:
      torch.nn.init.zeros_(self.bias)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''' Defines the octree convolution.

    Args:
      data (torch.Tensor): The input data.
      octree (Octree): The corresponding octree.
      depth (int): The depth of current octree.
    '''

    assert data.shape[1] == self.in_channels, 'Input channel mismatch.'

    if self.nempty:
      data = octree_pad(data, octree, depth)

    out = data.view(-1, 8 * self.in_channels) @ self.weights.flatten(0, 1)
    out = out.view(-1, self.out_channels)

    if not self.nempty:
      out = octree_pad(out, octree, depth-1)

    if self.use_bias:
      out += self.bias
    return out

  def extra_repr(self) -> str:
    r''' Sets the extra representation of the module.
    '''

    return ('in_channels={}, out_channels={}, kernel_size=2, stride=2, '
            'nempty={}, bias={}').format(self.in_channels, self.out_channels,
             self.nempty, self.use_bias)  # noqa


class OctreeConvK1(torch.nn.Module):
  r''' Performs octree convolution with kernel size 1 and stride 1.

  The shape of octree features is :obj:`(N, C)`, where :obj:`N` is the node
  number and :obj:`C` is the feature channel. Therefore, :class:`Conv1x1` can be
  implemented with :class:`torch.nn.Linear`.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               use_bias: bool = False, **kwargs):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.use_bias = use_bias

    self.linear = torch.nn.Linear(in_channels, out_channels, use_bias)

  def forward(self, data: torch.Tensor, **kwargs):
    r''''''

    return self.linear(data)

  def extra_repr(self) -> str:
    r''' Sets the extra representation of the module.
    '''

    return ('in_channels={}, out_channels={}, kernel_size=1, stride=1, '
            'bias={}').format(self.in_channels, self.out_channels, self.use_bias)
