import torch
import ocnn


class LeNet(torch.nn.Module):
  r''' Octree-based LeNet for classification.
  '''

  def __init__(self, in_channels: int, out_channels: int, stages: int,
               nempty: bool = False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stages = stages
    self.nempty = nempty
    channels = [in_channels] + [2 ** max(i+8-stages, 2) for i in range(stages)]

    self.convs = torch.nn.ModuleList(
        [ocnn.modules.OctreeConvBnRelu(channels[i], channels[i+1], nempty=nempty)
         for i in range(stages)])
    self.pools = torch.nn.ModuleList(
        [ocnn.nn.OctreeMaxPool(nempty) for i in range(stages)])
    self.header = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),                                   # drop1
        ocnn.modules.FcBnRelu(channels[3] * 64, channels[2]),      # fc1
        torch.nn.Dropout(p=0.5),                                   # drop2
        torch.nn.Linear(channels[2], out_channels))                # fc2

  def forward(self, octree: ocnn.octree.Octree):
    r''''''

    depth = octree.depth
    data = octree.get_input_feature()
    if not self.nempty:
      data = ocnn.nn.octree_pad(data, octree, depth)
    assert data.size(1) == self.in_channels

    for i in range(self.stages):
      d = depth - i
      data = self.convs[i](data, octree, d)
      data = self.pools[i](data, octree, d)
    data = ocnn.nn.octree2voxel(data, octree, depth-self.stages, self.nempty)
    data = self.header(data)
    return data
