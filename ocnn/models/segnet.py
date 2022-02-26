import torch
import ocnn


class SegNet(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int, stages: int,
               interp: str = 'linear', nempty: bool = False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stages = stages
    self.nempty = nempty
    return_indices = True

    channels = [in_channels] + [2 ** max(i+7-stages, 2) for i in range(stages)]
    # channels[2] = channels[3]

    self.input_feature = ocnn.modules.InputFeature(in_channels, nempty)
    self.convs = torch.nn.ModuleList(
        [ocnn.modules.OctreeConvBnRelu(channels[i], channels[i+1], nempty=nempty)
         for i in range(stages)])
    self.pools = torch.nn.ModuleList(
        [ocnn.nn.OctreeMaxPool(nempty, return_indices) for i in range(stages)])

    self.deconvs = torch.nn.ModuleList(
        [ocnn.modules.OctreeConvBnRelu(channels[i], channels[i-1], nempty=nempty)
         for i in range(stages-1, -1, -1)])
    self.unpools = torch.nn.ModuleList(
        [ocnn.nn.OctreeMaxUnpool(nempty) for i in range(stages)])
    self.deconv = ocnn.modules.OctreeConvBnRelu(channels[-1], channels[-1])
    self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)

    self.header = torch.nn.Sequential(
        ocnn.modules.Conv1x1BnRelu(channels[-1], 64),
        ocnn.modules.Conv1x1(64, out_channels, use_bias=True))

  def forward(self, octree: ocnn.octree.Octree, pts: torch.Tensor):
    r''''''

    depth = octree.depth
    data = self.input_feature(octree)

    # encoder
    indices = [None] * (depth + 1)
    for i in range(self.stages):
      d = depth - i
      data = self.convs[i](data, octree, d)
      data, indices[d] = self.pools[i](data, octree, d)

    # decoder
    for i in enumerate(self.stages):
      d = i + 2
      data = self.deconvs[i](data, octree, d)
      data = self.unpools[i](data, indices[d], octree, d)

    # point feature
    feature = self.deconv(data, octree)
    feature = self.octree_interp(feature, octree, pts, depth)

    # header
    logits = self.header(feature)
    return logits
