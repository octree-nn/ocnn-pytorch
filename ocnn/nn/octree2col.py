import torch
import torch.nn

from ocnn.octree import Octree
from ocnn.utils import scatter_add


def octree2col(data: torch.Tensor, octree: Octree, depth: int,
               kernel_size: str = '333', stride: int = 1, nempty: bool = False):
  r''' Gathers the neighboring features for convolutions.

  Args:
    data (torch.Tensor): The input data.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current octree.
    kernel_size (str): The kernel shape, choose from :obj:`333`, :obj:`311`, 
        :obj:`131`, :obj:`113`, :obj:`222`, :obj:`331`, :obj:`133`, and
        :obj:`313`.
    stride (int): The stride of neighborhoods (:obj:`1` or :obj:`2`). If the
        stride is :obj:`2`, it always returns the neighborhood of the first
        siblings, and the number of elements of output tensor is
        :obj:`octree.nnum[depth] / 8`.
    nempty (bool): If True, only returns the neighborhoods of the non-empty
        octree nodes.
  '''

  neigh = octree.get_neigh(depth, kernel_size, stride, nempty)
  size = (neigh.shape[0], neigh.shape[1], data.shape[1])
  out = torch.zeros(size, dtype=data.dtype, device=data.device)
  valid = neigh >= 0
  out[valid] = data[neigh[valid]]  # (N, K, C)
  return out


def col2octree(data: torch.Tensor, octree: Octree, depth: int,
               kernel_size: str = '333', stride: int = 1, nempty: bool = False):
  r''' Scatters the convolution features to an octree.

  Please refer to :func:`octree2col` for the usage of function parameters.
  '''

  neigh = octree.get_neigh(depth, kernel_size, stride, nempty)
  valid = neigh >= 0
  dim_size = octree.nnum_nempty[depth] if nempty else octree.nnum[depth]
  out = scatter_add(data[valid], neigh[valid], dim=0, dim_size=dim_size)
  return out
