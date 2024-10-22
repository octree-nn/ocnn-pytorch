# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.utils.checkpoint
from typing import List

from ocnn.nn import OctreeConv, OctreeDeconv, OctreeGroupNorm
from ocnn.octree import Octree


# bn_momentum, bn_eps = 0.01, 0.001  # the default value of Tensorflow 1.x
# bn_momentum, bn_eps = 0.1, 1e-05   # the default value of pytorch


def ckpt_conv_wrapper(conv_op, data, octree):
  # The dummy tensor is a workaround when the checkpoint is used for the first conv layer:
  # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
  dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

  def conv_wrapper(data, octree, dummy_tensor):
    return conv_op(data, octree)

  return torch.utils.checkpoint.checkpoint(conv_wrapper, data, octree, dummy)


class OctreeConvBn(torch.nn.Module):
  r''' A sequence of :class:`OctreeConv` and :obj:`BatchNorm`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm1d(out_channels)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.bn(out)
    return out


class OctreeConvBnRelu(torch.nn.Module):
  r''' A sequence of :class:`OctreeConv`, :obj:`BatchNorm`, and :obj:`Relu`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm1d(out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.bn(out)
    out = self.relu(out)
    return out


class OctreeDeconvBnRelu(torch.nn.Module):
  r''' A sequence of :class:`OctreeDeconv`, :obj:`BatchNorm`, and :obj:`Relu`.

  Please refer to :class:`ocnn.nn.OctreeDeconv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.deconv = OctreeDeconv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm1d(out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.deconv(data, octree, depth)
    out = self.bn(out)
    out = self.relu(out)
    return out


class Conv1x1(torch.nn.Module):
  r''' Performs a convolution with kernel :obj:`(1,1,1)`.

  The shape of octree features is :obj:`(N, C)`, where :obj:`N` is the node
  number and :obj:`C` is the feature channel. Therefore, :class:`Conv1x1` can be
  implemented with :class:`torch.nn.Linear`.
  '''

  def __init__(self, in_channels: int, out_channels: int, use_bias: bool = False):
    super().__init__()
    self.linear = torch.nn.Linear(in_channels, out_channels, use_bias)

  def forward(self, data: torch.Tensor):
    r''''''

    return self.linear(data)


class Conv1x1Bn(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1` and :class:`BatchNorm`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.bn = torch.nn.BatchNorm1d(out_channels)

  def forward(self, data: torch.Tensor):
    r''''''

    out = self.conv(data)
    out = self.bn(out)
    return out


class Conv1x1BnRelu(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1`, :class:`BatchNorm` and :class:`Relu`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.bn = torch.nn.BatchNorm1d(out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor):
    r''''''

    out = self.conv(data)
    out = self.bn(out)
    out = self.relu(out)
    return out


class FcBnRelu(torch.nn.Module):
  r''' A sequence of :class:`FC`, :class:`BatchNorm` and :class:`Relu`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.flatten = torch.nn.Flatten(start_dim=1)
    self.fc = torch.nn.Linear(in_channels, out_channels, bias=False)
    self.bn = torch.nn.BatchNorm1d(out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data):
    r''''''

    out = self.flatten(data)
    out = self.fc(out)
    out = self.bn(out)
    out = self.relu(out)
    return out


class OctreeConvGn(torch.nn.Module):
  r''' A sequence of :class:`OctreeConv` and :obj:`OctreeGroupNorm`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int, group: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.gn = OctreeGroupNorm(out_channels, group=group, nempty=nempty)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.gn(out, octree, depth)
    return out


class OctreeConvGnRelu(torch.nn.Module):
  r''' A sequence of :class:`OctreeConv`, :obj:`OctreeGroupNorm`, and :obj:`Relu`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int, group: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.stride = stride
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.gn = OctreeGroupNorm(out_channels, group=group, nempty=nempty)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.gn(out, octree, depth if self.stride == 1 else depth - 1)
    out = self.relu(out)
    return out


class OctreeDeconvGnRelu(torch.nn.Module):
  r''' A sequence of :class:`OctreeDeconv`, :obj:`OctreeGroupNorm`, and :obj:`Relu`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int, group: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.stride = stride
    self.deconv = OctreeDeconv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.gn = OctreeGroupNorm(out_channels, group=group, nempty=nempty)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.deconv(data, octree, depth)
    out = self.gn(out, octree, depth if self.stride == 1 else depth + 1)
    out = self.relu(out)
    return out


class Conv1x1Gn(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1`, :class:`OctreeGroupNorm`.
  '''

  def __init__(self, in_channels: int, out_channels: int, group: int,
               nempty: bool = False):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.gn = OctreeGroupNorm(out_channels, group=group, nempty=nempty)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data)
    out = self.gn(out, octree, depth)
    return out


class Conv1x1GnRelu(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1`, :class:`OctreeGroupNorm` and :class:`Relu`.
  '''

  def __init__(self, in_channels: int, out_channels: int, group: int,
               nempty: bool = False):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.gn = OctreeGroupNorm(out_channels, group=group, nempty=nempty)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data)
    out = self.gn(out, octree, depth)
    out = self.relu(out)
    return out


class InputFeature(torch.nn.Module):
  r''' Returns the initial input feature stored in octree.

  Refer to :func:`ocnn.octree.Octree.get_input_feature` for details.
  '''

  def __init__(self, feature: str = 'NDF', nempty: bool = False):
    super().__init__()
    self.nempty = nempty
    self.feature = feature.upper()

  def forward(self, octree: Octree):
    r''''''
    return octree.get_input_feature(self.feature, self.nempty)

  def extra_repr(self) -> str:
    r''''''
    return 'feature={}, nempty={}'.format(self.feature, self.nempty)
