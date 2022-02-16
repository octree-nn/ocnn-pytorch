import torch
import torch.nn

from ocnn.octree import Octree


class Octree2Col(torch.nn.Module):
  r''' Octree2Col. 
  '''

  def __init__(self, kernel_size: str = '333', stride: int = 1, nempty: bool = False):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.nempty = nempty

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    neigh = octree.get_neigh(depth, self.kernel_size, self.stride, self.nempty)
    out = data[neigh]  # (N, K, C)
    return out

  def extra_repr(self) -> str:
    return 'kernel_size={}, stride={}, nempty={}'.format(
        self.kernel_size, self.stride, self.nempty)
