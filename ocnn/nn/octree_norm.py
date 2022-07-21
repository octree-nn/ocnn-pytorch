import torch
import torch.nn

from ocnn.octree import Octree
from ocnn.utils import scatter_add


OctreeBatchNorm = torch.nn.BatchNorm1d


class OctreeInstanceNorm(torch.nn.Module):
  r''' An instance normalization layer for the octree.
  '''

  def __init__(self, in_channels: int, nempty: bool = False):
    super().__init__()

    self.eps = 1e-5
    self.nempty = nempty
    self.in_channels = in_channels

    self.weights = torch.nn.Parameter(torch.Tensor(1, in_channels))
    self.bias = torch.nn.Parameter(torch.Tensor(1, in_channels))
    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.ones_(self.weights)
    torch.nn.init.zeros_(self.bias)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    batch_size = octree.batch_size
    batch_id = octree.batch_id(depth, self.nempty)
    ones = data.new_ones([data.shape[0], 1])
    count = scatter_add(ones, batch_id, dim=0, dim_size=batch_size)
    norm = 1.0 / (count + self.eps)  # there might be 0 element in some shapes

    mean = scatter_add(data, batch_id, dim=0, dim_size=batch_size) * norm
    out = data - mean[batch_id]
    var = scatter_add(out * out, batch_id, dim=0, dim_size=batch_size) * norm
    inv_std = 1.0 / (var + self.eps).sqrt()
    out = out * inv_std[batch_id]

    out = out * self.weights + self.bias
    return out

  def extra_repr(self) -> str:
    return ('in_channels={}, nempty={}').format(self.in_channels, self.nempty)
