# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn
from typing import Optional
from torchvision.models import resnet18

import ocnn
from ocnn.octree import Octree


class Image2Shape(torch.nn.Module):
  r''' Octree-based AutoEncoder for shape encoding and decoding.

  Args:
    channel_out (int): The channel of the output signal.
    depth (int): The depth of the octree.
    full_depth (int): The full depth of the octree.
  '''

  def __init__(self, channel_out: int, depth: int, full_depth: int = 2,
               code_channel: int = 32):
    super().__init__()
    self.depth = depth
    self.full_depth = full_depth
    self.channel_out = channel_out
    self.resblk_num = 2
    self.channels = [512, 512, 256, 256, 128, 128, 64, 64, 32, 32]
    self.code_channel = code_channel

    # encoder
    self.resnet18 = resnet18()
    channel = self.code_channel * 2 ** (3 * full_depth)
    # self.resnet18.fc = ocnn.modules.Conv1x1BnRelu(512, channel)
    self.resnet18.fc = torch.nn.Linear(512, channel, bias=True)

    # decoder
    self.channels[full_depth] = self.code_channel  # update `channels`
    self.upsample = torch.nn.ModuleList([ocnn.modules.OctreeDeconvBnRelu(
        self.channels[d-1], self.channels[d], kernel_size=[2], stride=2,
        nempty=False) for d in range(full_depth+1, depth+1)])
    self.decoder_blks = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
        self.channels[d], self.channels[d], self.resblk_num, nempty=False)
        for d in range(full_depth, depth+1)])

    # header
    self.predict = torch.nn.ModuleList([self._make_predict_module(
        self.channels[d], 2) for d in range(full_depth, depth + 1)])
    self.header = self._make_predict_module(self.channels[depth], channel_out)

  def _make_predict_module(self, channel_in, channel_out=2, num_hidden=64):
    return torch.nn.Sequential(
        ocnn.modules.Conv1x1BnRelu(channel_in, num_hidden),
        ocnn.modules.Conv1x1(num_hidden, channel_out, use_bias=True))

  def decoder(self, shape_code: torch.Tensor, octree: Octree,
              update_octree: bool = False):
    r''' The decoder network of the AutoEncoder.
    '''

    logits = dict()
    deconv = shape_code
    depth, full_depth = self.depth, self.full_depth
    for i, d in enumerate(range(full_depth, depth+1)):
      if d > full_depth:
        deconv = self.upsample[i-1](deconv, octree, d-1)
      deconv = self.decoder_blks[i](deconv, octree, d)

      # predict the splitting label
      logit = self.predict[i](deconv)
      logits[d] = logit

      # update the octree according to predicted labels
      if update_octree:
        split = logit.argmax(1).int()
        octree.octree_split(split, d)
        if d < depth:
          octree.octree_grow(d + 1)

      # predict the signal
      if d == depth:
        signal = self.header(deconv)
        signal = torch.tanh(signal)
        signal = ocnn.nn.octree_depad(signal, octree, depth)
        if update_octree:
          octree.features[depth] = signal

    return {'logits': logits, 'signal': signal, 'octree_out': octree}

  def decode_code(self, shape_code: torch.Tensor):
    r''' Decodes the shape code to an output octree.

    Args:
      shape_code (torch.Tensor): The shape code for decoding.
    '''

    octree_out = self.init_octree(shape_code)
    out = self.decoder(shape_code, octree_out, update_octree=True)
    return out

  def init_octree(self, shape_code: torch.Tensor):
    r''' Initialize a full octree for decoding.

    Args:
      shape_code (torch.Tensor): The shape code for decoding, used to getting
          the `batch_size` and `device` to initialize the output octree.
    '''

    node_num = 2 ** (3 * self.full_depth)
    batch_size = shape_code.size(0) // node_num
    octree = ocnn.octree.init_octree(
        self.depth, self.full_depth, batch_size, shape_code.device)
    return octree

  def forward(self, image: torch.Tensor, octree: Optional[Octree] = None,
              update_octree: bool = False):
    r''''''

    shape_code = self.resnet18(image)
    shape_code = shape_code.view(-1, self.code_channel)
    if update_octree:
      octree = self.init_octree(shape_code)
    out = self.decoder(shape_code, octree, update_octree)
    return out
