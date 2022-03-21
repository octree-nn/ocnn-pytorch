import torch
import ocnn
from ocnn.octree import Octree


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
    channels = [in_channels] + [2 ** max(i+7-stages, 2) for i in range(stages)]

    self.convs = torch.nn.ModuleList(
        [ocnn.modules.OctreeConvBnRelu(channels[i], channels[i+1], nempty=nempty)
         for i in range(stages)])
    self.pools = torch.nn.ModuleList(
        [ocnn.nn.OctreeMaxPool(nempty) for i in range(stages)])
    self.octree2voxel = ocnn.nn.Octree2Voxel(self.nempty)
    self.header = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),                     # drop1
        ocnn.modules.FcBnRelu(64 * 64, 128),         # fc1
        torch.nn.Dropout(p=0.5),                     # drop2
        torch.nn.Linear(128, out_channels))          # fc2

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    for i in range(self.stages):
      d = depth - i
      data = self.convs[i](data, octree, d)
      data = self.pools[i](data, octree, d)
    data = self.octree2voxel(data, octree, depth-self.stages)
    data = self.header(data)
    return data
