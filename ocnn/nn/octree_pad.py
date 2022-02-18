import torch

from ocnn.octree import Octree


def octree_pad(data: torch.Tensor, octree: Octree, depth: int):
  r''' Pads zeros to make the number of elements of :attr:`data` equal to the
  octree node number.

  Args:
    data (torch.Tensor): The input tensor with its number of elements equal to the
        non-empty octree node number.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current octree.
  '''

  child = octree.children[depth]
  mask = child >= 0
  size = (octree.nnum[depth], data.shape[1])  # (N, C)
  out = torch.zeros(size, dtype=data.dtype, device=data.device)
  out[mask] = data
  return out


def octree_depad(data: torch.Tensor, octree: Octree, depth: int):
  r''' Reverse operation of :func:`octree_depad`.

  Please refer to :func:`octree_depad` for the meaning of the arguments.
  '''

  child = octree.children[depth]
  mask = child >= 0
  return data[mask]