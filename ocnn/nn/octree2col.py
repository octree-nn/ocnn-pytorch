import torch
import torch.nn

from ocnn.octree import Octree


class Octree2Col(torch.nn.Module):
  r''' Gathers the neighboring features for convolutions.

  Args:
    kernel_size (str): The kernel shape from :obj:`333`, :obj:`311`, :obj:`131`,
        :obj:`113`, :obj:`222`, :obj:`331`, :obj:`133`, and :obj:`313`.
    stride (int): The stride of neighborhoods (:obj:`1` or :obj:`2`). If the
        stride is :obj:`2`, always returns the neighborhood of the first
        siblings.
    nempty (bool): If True, only returns the neighborhoods of the non-empty
        octree nodes.
    '''

  def __init__(self, kernel_size: str = '333', stride: int = 1, nempty: bool = False):
    super().__init__()

    self.kernel_size = kernel_size
    self.stride = stride
    self.nempty = nempty

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''' Defines the forward computation.

    Args:
      data (torch.Tensor): The input data.
      octree (Octree): The corresponding octree.
      depth (int): The depth of current octree.
    '''

    neigh = octree.get_neigh(depth, self.kernel_size, self.stride, self.nempty)
    size = (neigh.shape[0], neigh.shape[1], data.shape[1])
    out = torch.zeros(size, dtype=data.dtype, device=data.device)
    valid = neigh >= 0
    out[valid] = data[neigh[valid]]  # (N, K, C)
    return out

  def extra_repr(self) -> str:
    r''' Sets the extra representation of the module.
    '''

    return 'kernel_size={}, stride={}, nempty={}'.format(
        self.kernel_size, self.stride, self.nempty)
