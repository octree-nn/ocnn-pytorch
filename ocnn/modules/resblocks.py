# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.utils.checkpoint

from ocnn.octree import Octree
from ocnn.nn import OctreeMaxPool
from ocnn.modules import Conv1x1BnRelu, OctreeConvBnRelu, Conv1x1Bn, OctreeConvBn


class OctreeResBlock(torch.nn.Module):
  r''' Octree-based ResNet block in a bottleneck style. The block is composed of
  a series of :obj:`Conv1x1`, :obj:`Conv3x3`, and :obj:`Conv1x1`.

  Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    stride (int): The stride of the block (:obj:`1` or :obj:`2`).
    bottleneck (int): The input and output channels of the :obj:`Conv3x3` is
        equal to the input channel divided by :attr:`bottleneck`.
    nempty (bool): If True, only performs the convolution on non-empty
        octree nodes.
  '''

  def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
               bottleneck: int = 4, nempty: bool = False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.bottleneck = bottleneck
    self.stride = stride
    channelb = int(out_channels / bottleneck)

    if self.stride == 2:
      self.max_pool = OctreeMaxPool(nempty)
    self.conv1x1a = Conv1x1BnRelu(in_channels, channelb)
    self.conv3x3 = OctreeConvBnRelu(channelb, channelb, nempty=nempty)
    self.conv1x1b = Conv1x1Bn(channelb, out_channels)
    if self.in_channels != self.out_channels:
      self.conv1x1c = Conv1x1Bn(in_channels, out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    if self.stride == 2:
      data = self.max_pool(data, octree, depth)
      depth = depth - 1
    conv1 = self.conv1x1a(data)
    conv2 = self.conv3x3(conv1, octree, depth)
    conv3 = self.conv1x1b(conv2)
    if self.in_channels != self.out_channels:
      data = self.conv1x1c(data)
    out = self.relu(conv3 + data)
    return out


class OctreeResBlock2(torch.nn.Module):
  r''' Basic Octree-based ResNet block. The block is composed of
  a series of :obj:`Conv3x3` and :obj:`Conv3x3`.

  Refer to :class:`OctreeResBlock` for the details of arguments.
  '''

  def __init__(self, in_channels, out_channels, stride=1, bottleneck=1,
               nempty=False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride
    channelb = int(out_channels / bottleneck)

    if self.stride == 2:
      self.maxpool = OctreeMaxPool(self.depth)
    self.conv3x3a = OctreeConvBnRelu(in_channels, channelb, nempty=nempty)
    self.conv3x3b = OctreeConvBn(channelb, out_channels, nempty=nempty)
    if self.in_channels != self.out_channels:
      self.conv1x1 = Conv1x1Bn(in_channels, out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    if self.stride == 2:
      data = self.maxpool(data, octree, depth)
      depth = depth - 1
    conv1 = self.conv3x3a(data, octree, depth)
    conv2 = self.conv3x3b(conv1, octree, depth)
    if self.in_channels != self.out_channels:
      data = self.conv1x1(data)
    out = self.relu(conv2 + data)
    return out


class OctreeResBlocks(torch.nn.Module):
  r''' A sequence of :attr:`resblk_num` ResNet blocks.
  '''

  def __init__(self, in_channels, out_channels, resblk_num, bottleneck=4,
               nempty=False, resblk=OctreeResBlock, use_checkpoint=False):
    super().__init__()
    self.resblk_num = resblk_num
    self.use_checkpoint = use_checkpoint
    channels = [in_channels] + [out_channels] * resblk_num

    self.resblks = torch.nn.ModuleList(
        [resblk(channels[i], channels[i+1], 1, bottleneck, nempty)
         for i in range(self.resblk_num)])

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    for i in range(self.resblk_num):
      if self.use_checkpoint:
        data = torch.utils.checkpoint.checkpoint(
            self.resblks[i], data, octree, depth)
      else:
        data = self.resblks[i](data, octree, depth)
    return data
