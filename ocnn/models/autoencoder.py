import torch
import torch.nn
from typing import Optional

import ocnn
from ocnn.octree import Octree


class AutoEncoder(torch.nn.Module):
  r''' Octree-based AutoEncoder for shape encoding and decoding.

  Args:
    channel_in (int): The channel of the input signal.
    channel_out (int): The channel of the output signal.
    depth (int): The depth of the octree.
    full_depth (int): The full depth of the octree.
    feature (str): The feature type of the input signal. For details of this
        argument, please refer to :class:`ocnn.modules.InputFeature`.
  '''

  def __init__(self, channel_in: int, channel_out: int, depth: int,
               full_depth: int = 2, feature: str = 'ND'):
    super().__init__()
    self.channel_in = channel_in
    self.channel_out = channel_out
    self.depth = depth
    self.full_depth = full_depth
    self.feature = feature
    self.resblk_num = 2
    self.shape_code_channel = 128
    self.channels = [512, 512, 256, 256, 128, 128, 32, 32, 16, 16]

    # encoder
    self.conv1 = ocnn.modules.OctreeConvBnRelu(
        channel_in, self.channels[depth], nempty=False)
    self.encoder_blks = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
        self.channels[d], self.channels[d], self.resblk_num, nempty=False)
        for d in range(depth, full_depth-1, -1)])
    self.downsample = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
        self.channels[d], self.channels[d-1], kernel_size=[2], stride=2,
        nempty=False) for d in range(depth, full_depth, -1)])
    self.proj = torch.nn.Linear(
        self.channels[full_depth], self.shape_code_channel, bias=True)

    # decoder
    self.channels[full_depth] = self.shape_code_channel  # update `channels`
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

  def get_input_feature(self, octree: Octree):
    r''' Get the input feature from the input `octree`.
    '''

    octree_feature = ocnn.modules.InputFeature(self.feature, nempty=False)
    out = octree_feature(octree)
    assert out.size(1) == self.channel_in
    return out

  def ae_encoder(self, octree: Octree):
    r''' The encoder network of the AutoEncoder.
    '''

    convs = dict()
    depth, full_depth = self.depth, self.full_depth
    data = self.get_input_feature(octree)
    convs[depth] = self.conv1(data, octree, depth)
    for i, d in enumerate(range(depth, full_depth-1, -1)):
      convs[d] = self.encoder_blks[i](convs[d], octree, d)
      if d > full_depth:
        convs[d-1] = self.downsample[i](convs[d], octree, d)

    # NOTE: here tanh is used to constrain the shape code in [-1, 1]
    shape_code = self.proj(convs[full_depth]).tanh()
    return shape_code

  def ae_decoder(self, shape_code: torch.Tensor, octree: Octree,
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
    out = self.ae_decoder(shape_code, octree_out, update_octree=True)
    return out

  def init_octree(self, shape_code: torch.Tensor):
    r''' Initialize a full octree for decoding.

    Args:
      shape_code (torch.Tensor): The shape code for decoding, used to getting 
          the `batch_size` and `device` to initialize the output octree.
    '''

    device = shape_code.device
    node_num = 2 ** (3 * self.full_depth)
    batch_size = shape_code.size(0) // node_num
    octree = Octree(self.depth, self.full_depth, batch_size, device)
    for d in range(self.full_depth+1):
      octree.octree_grow_full(depth=d)
    return octree

  def forward(self, octree: Octree, update_octree: bool):
    r''''''

    shape_code = self.ae_encoder(octree)
    if update_octree:
      octree = self.init_octree(shape_code)
    out = self.ae_decoder(shape_code, octree, update_octree)
    return out
