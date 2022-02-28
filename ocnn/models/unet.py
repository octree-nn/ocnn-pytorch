import torch
import torch.nn
from typing import Dict

import ocnn
from ocnn.octree import Octree


class UNet(torch.nn.Module):
  r''' Octree-based UNet for segmentation
  '''

  def __init__(self, in_channels: int, out_channels: int, interp: str = 'linear',
               nempty: bool = False):
    super(UNet, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.nempty = nempty
    self.config_network()
    self.encoder_stages = len(self.encoder_blocks)
    self.decoder_stages = len(self.decoder_blocks)

    # encoder
    self.input_feature = ocnn.modules.InputFeature(in_channels, nempty)
    self.conv1 = ocnn.modules.OctreeConvBnRelu(
        in_channels, self.encoder_channel[0], nempty=nempty)
    self.downsample = torch.nn.ModuleList(
        [ocnn.modules.OctreeConvBnRelu(self.encoder_channel[i],
         self.encoder_channel[i+1], kernel_size=[2], stride=2, nempty=nempty)
         for i in range(self.encoder_stages)])
    self.encoder = torch.nn.ModuleList(
        [ocnn.modules.OctreeResBlocks(self.encoder_channel[i+1],
         self.encoder_channel[i+1], self.encoder_blocks[i], self.bottleneck,
         nempty, self.resblk) for i in range(self.encoder_stages)])

    # decoder
    channel = [self.decoder_channel[i+1] + self.encoder_channel[-i-2]
               for i in range(self.decoder_stages)]
    self.upsample = torch.nn.ModuleList(
        [ocnn.modules.OctreeDeconvBnRelu(self.decoder_channel[i],
         self.decoder_channel[i+1], kernel_size=[2], stride=2, nempty=nempty)
         for i in range(self.decoder_stages)])
    self.decoder = torch.nn.ModuleList(
        [ocnn.modules.OctreeResBlocks(channel[i],
         self.decoder_channel[i+1], self.decoder_blocks[i], self.bottleneck,
         nempty, self.resblk) for i in range(self.decoder_stages)])

    # header
    # channel = self.decoder_channel[self.decoder_stages]
    self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)
    self.header = torch.nn.Sequential(
        ocnn.modules.Conv1x1BnRelu(self.decoder_channel[-1], 64),
        ocnn.modules.Conv1x1(64, self.out_channels, use_bias=True))

  def config_network(self):
    r''' Configure the network channels and Resblock numbers.
    '''

    self.encoder_channel = [32, 32, 64, 128, 256]
    self.decoder_channel = [256, 256, 128, 96, 96]
    self.encoder_blocks = [2, 3, 4, 6]
    self.decoder_blocks = [2, 2, 2, 2]
    self.bottleneck = 1
    self.resblk = ocnn.modules.OctreeResBlock2

  def unet_encoder(self, octree: Octree):
    r''' The encoder of the U-Net. 
    '''

    depth = octree.depth
    data = self.input_feature(octree)

    convd = dict()
    convd[depth] = self.conv1(data, octree, depth)
    for i in range(self.encoder_stages):
      d = depth - i
      conv = self.downsample[i](convd[d], octree, d)
      convd[d-1] = self.encoder[i](conv, octree, d-1)
    return convd

  def unet_decoder(self, octree: Octree, convd: Dict[int, torch.Tensor]):
    r''' The decoder of the U-Net. 
    '''

    depth = octree.depth - self.encoder_stages
    deconv = convd[depth]

    for i in range(self.decoder_stages):
      d = depth + i
      deconv = self.upsample[i](deconv, octree, d)
      deconv = torch.cat([convd[d+1], deconv], dim=1)  # skip connections
      deconv = self.decoder[i](deconv, octree, d+1)
    return deconv

  def forward(self, octree: Octree, pts: torch.Tensor):
    r''''''

    convd = self.unet_encoder(octree)
    deconv = self.unet_decoder(octree, convd)

    d = octree.depth - self.encoder_stages + self.decoder_stages
    feature = self.octree_interp(deconv, octree, d, pts)
    logits = self.header(feature)
    return logits
