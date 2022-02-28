import torch
import ocnn


class SegNet(torch.nn.Module):
  r''' Octree-based SegNet for segmentation.
  '''

  def __init__(self, in_channels: int, out_channels: int, stages: int,
               interp: str = 'linear', nempty: bool = False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stages = stages
    self.nempty = nempty
    return_indices = True

    channels_stages = [2 ** max(i+8-stages, 2) for i in range(stages)]
    channels = [in_channels] + channels_stages
    self.input_feature = ocnn.modules.InputFeature(in_channels, nempty)
    self.convs = torch.nn.ModuleList(
        [ocnn.modules.OctreeConvBnRelu(channels[i], channels[i+1], nempty=nempty)
         for i in range(stages)])
    self.pools = torch.nn.ModuleList(
        [ocnn.nn.OctreeMaxPool(nempty, return_indices) for i in range(stages)])

    self.bottleneck = ocnn.modules.OctreeConvBnRelu(channels[-1], channels[-1])

    channels = channels_stages[::-1] + [channels_stages[0]]
    self.deconvs = torch.nn.ModuleList(
        [ocnn.modules.OctreeConvBnRelu(channels[i], channels[i+1], nempty=nempty)
         for i in range(0, stages)])
    self.unpools = torch.nn.ModuleList(
        [ocnn.nn.OctreeMaxUnpool(nempty) for i in range(stages)])

    self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)
    self.header = torch.nn.Sequential(
        ocnn.modules.Conv1x1BnRelu(channels[-1], 64),
        ocnn.modules.Conv1x1(64, out_channels, use_bias=True))

  def forward(self, octree: ocnn.octree.Octree, pts: torch.Tensor):
    r''''''

    depth = octree.depth
    data = self.input_feature(octree)

    # encoder
    indices = dict()
    for i in range(self.stages):
      d = depth - i
      data = self.convs[i](data, octree, d)
      data, indices[d] = self.pools[i](data, octree, d)

    # bottleneck
    data = self.bottleneck(data, octree, 2)

    # decoder
    for i in range(self.stages):
      d = i + 3
      data = self.unpools[i](data, indices[d], octree, d-1)
      data = self.deconvs[i](data, octree, d)

    # header
    feature = self.octree_interp(data, octree, depth, pts)
    logits = self.header(feature)

    return logits
