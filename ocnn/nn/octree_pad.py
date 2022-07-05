import torch

from ..octree import Octree


def octree_pad(data: torch.Tensor, octree: Octree, depth: int, val: float = 0.0):
  r''' Pads :attr:`val` to make the number of elements of :attr:`data` equal to
  the octree node number.

  Args:
    data (torch.Tensor): The input tensor with its number of elements equal to the
        non-empty octree node number.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current octree.
    val (float): The padding value. (Default: :obj:`0.0`)
  '''

  mask = octree.nempty_mask(depth)
  size = (octree.nnum[depth], data.shape[1])  # (N, C)
  out = torch.full(size, val, dtype=data.dtype, device=data.device)
  out[mask] = data
  return out


def octree_depad(data: torch.Tensor, octree: Octree, depth: int):
  r''' Reverse operation of :func:`octree_depad`.

  Please refer to :func:`octree_depad` for the meaning of the arguments.
  '''

  mask = octree.nempty_mask(depth)
  return data[mask]
