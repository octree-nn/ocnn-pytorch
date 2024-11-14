# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------


import ocnn
import torch
import torch.nn

from ocnn.octree import Octree
from ocnn.models.autoencoder import AutoEncoder


class OUNet(AutoEncoder):

  def __init__(self, channel_in: int, channel_out: int, depth: int,
               full_depth: int = 2, feature: str = 'ND'):
    super().__init__(channel_in, channel_out, depth, full_depth, feature)
    self.proj = None  # remove this module used in AutoEncoder

  def encoder(self, octree):
    r''' The encoder network for extracting heirarchy features.
    '''

    convs = dict()
    depth, full_depth = self.depth, self.full_depth
    data = octree.get_input_feature(self.feature, nempty=False)
    assert data.size(1) == self.channel_in
    convs[depth] = self.conv1(data, octree, depth)
    for i, d in enumerate(range(depth, full_depth-1, -1)):
      convs[d] = self.encoder_blks[i](convs[d], octree, d)
      if d > full_depth:
        convs[d-1] = self.downsample[i](convs[d], octree, d)
    return convs

  def decoder(self, convs: dict, octree_in: Octree, octree_out: Octree,
              update_octree: bool = False):
    r''' The decoder network for decode the octree.
    '''

    logits = dict()
    deconv = convs[self.full_depth]
    depth, full_depth = self.depth, self.full_depth
    for i, d in enumerate(range(full_depth, depth + 1)):
      if d > full_depth:
        deconv = self.upsample[i-1](deconv, octree_out, d-1)
        skip = ocnn.nn.octree_align(convs[d], octree_in, octree_out, d)
        deconv = deconv + skip  # output-guided skip connections
      deconv = self.decoder_blks[i](deconv, octree_out, d)

      # predict the splitting label
      logit = self.predict[i](deconv)
      logits[d] = logit

      # update the octree according to predicted labels
      if update_octree:
        split = logit.argmax(1).int()
        octree_out.octree_split(split, d)
        if d < depth:
          octree_out.octree_grow(d + 1)

      # predict the signal
      if d == depth:
        signal = self.header(deconv)
        signal = torch.tanh(signal)
        signal = ocnn.nn.octree_depad(signal, octree_out, depth)
        if update_octree:
          octree_out.features[depth] = signal

    return {'logits': logits, 'signal': signal, 'octree_out': octree_out}

  def init_octree(self, octree_in: Octree):
    r''' Initialize a full octree for decoding.
    '''

    device = octree_in.device
    batch_size = octree_in.batch_size
    octree = Octree(self.depth, self.full_depth, batch_size, device)
    for d in range(self.full_depth+1):
      octree.octree_grow_full(depth=d)
    return octree

  def forward(self, octree_in, octree_out=None, update_octree: bool = False):
    r''''''

    if octree_out is None:
      update_octree = True
      octree_out = self.init_octree(octree_in)
    convs = self.encoder(octree_in)
    out = self.decoder(convs, octree_in, octree_out, update_octree)
    return out
