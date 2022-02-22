import torch
import torch.nn

from ..octree import Octree
from .octree_pad import octree_pad, octree_depad


def octree_max_pool(data: torch.Tensor, octree: Octree, depth: int,
                    nempty: bool = False, return_indices: bool = False):
  r''' Performs octree max pooling.

  Args:
    data (torch.Tensor): The input tensor.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current octree. After pooling, the corresponding
        depth decreased by 1.
    nempty (bool): If True, :attr:`data` contains only features of non-empty
        octree nodes.
    return_indices (bool): If True, returns the indices, which can be used in
        :func:`octree_max_unpool`.
  '''

  if nempty:
    data = octree_pad(data, octree, depth, float('-inf'))
  data = data.view(-1, 8, data.shape[1])
  out, indices = data.max(dim=1)
  if not nempty:
    out = octree_pad(out, octree, depth-1)
  return (out, indices) if return_indices else out


def octree_max_unpool(data: torch.Tensor, indices: torch.Tensor, octree: Octree,
                      depth: int, nempty: bool = False):
  r''' Performs octree max unpooling.

  Args:
    data (torch.Tensor): The input tensor.
    indices (torch.Tensor): The indices returned by :func:`octree_max_pool`.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current octree. After unpooling, the corresponding
        depth increases by 1.
  '''

  if not nempty:
    data = octree_depad(data, octree, depth)
  num, channel = data.shape
  out = torch.zeros(num, 8, channel, dtype=data.dtype, device=data.device)
  i = torch.arange(num, dtype=indices.dtype, device=indices.device)
  k = torch.arange(channel, dtype=indices.dtype, device=indices.device)
  i, k = torch.meshgrid(i, k, indexing='ij')
  out[i, indices, k] = data
  out = out.view(-1, channel)
  if nempty:
    out = octree_depad(out, octree, depth+1)
  return out


def octree_global_pool(data: torch.Tensor, octree: Octree, depth: int):
  r''' Performs octree global average pooling.

  Args:
    data (torch.Tensor): The input tensor.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current octree. After pooling, the corresponding
        depth decreased by 1.

  .. note::
    Currently, the :attr:`depth` must NOT be larger than :obj:`octree.full_depth`.
  '''

  torch._assert(depth <= octree.full_depth,
                'The depth = {} is too large.'.format(depth))
  nnum = octree.nnum[depth]
  out = data.view(-1, nnum, data.shape[1])
  return out.mean(dim=1)


class OctreeMaxPool(torch.nn.Module):
  r''' Performs octree max pooling.

  Please refer to :func:`octree_max_pool` for details.
  '''

  def __init__(self, nempty: bool = False, return_indices: bool = False):
    super().__init__()
    self.nempty = nempty
    self.return_indices = return_indices

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    return octree_max_pool(data, octree, depth, self.nempty, self.return_indices)


class OctreeMaxUnpool(torch.nn.Module):
  r''' Performs octree max unpooling.

  Please refer to :func:`octree_max_unpool` for details.
  '''

  def __init__(self, nempty: bool = False):
    super().__init__()
    self.nempty = nempty

  def forward(self, data: torch.Tensor, indices: torch.Tensor, octree: Octree,
              depth: int):
    r''''''

    return octree_max_unpool(data, indices, octree, depth, self.nempty)
