import torch
import ocnn


class ResNet(torch.nn.Module):
  r''' Octree-based ResNet for classification.
  '''

  def __init__(self, in_channels: int, out_channels: int, resblock_num: int,
               stages: int, nempty: bool = False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.resblk_num = resblock_num
    self.stages = stages
    self.nempty = nempty
    channels = [2 ** max(i+8-stages, 2) for i in range(stages+1)]

    self.conv1 = ocnn.modules.OctreeConvBnRelu(
        in_channels, channels[0], nempty=nempty)
    self.resblocks = torch.nn.ModuleList([
        ocnn.modules.OctreeResBlocks(channels[i], channels[i+1], resblock_num,
        nempty=nempty) for i in range(stages)])  # noqa
    self.pools = torch.nn.ModuleList(
        [ocnn.nn.OctreeMaxPool(nempty) for i in range(stages)])
    self.global_pool = ocnn.nn.OctreeGlobalPool()
    self.header = torch.nn.Linear(channels[-1], out_channels, bias=True)

  def forward(self, octree: ocnn.octree.Octree):
    r''''''

    depth = octree.depth
    data = octree.get_input_feature()
    if not self.nempty:
      data = ocnn.nn.octree_pad(data, octree, depth)
    assert data.size(1) == self.in_channels

    data = self.conv1(data, octree, depth)
    for i in range(self.stages):
      d = depth - i
      data = self.resblocks[i](data, octree, d)
      data = self.pools[i](data, octree, d)
    data = self.global_pool(data, octree, depth-self.stages)
    data = self.header(data)
    return data
