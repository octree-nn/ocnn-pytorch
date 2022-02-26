import torch
import torch.sparse
from typing import List

import ocnn
from ocnn.octree import Octree


def octree_nearest_pts(data: torch.Tensor, octree: Octree, depth: int,
                       pts: torch.Tensor, nempty: bool = False):
  ''' The nearest-neighbor interpolatation with input points.

  Args:
    data (torch.Tensor): The input data.
    octree (Octree): The octree to interpolate.
    depth (int): The depth of the data.
    pts (torch.Tensor): The coordinates of the points with shape :obj:`(N, 4)`,
        i.e. :obj:`N x (x, y, z, batch)`.
    nempty (bool): If true, the :attr:`data` only contains features of non-empty 
        octree nodes

    .. note::
    The :attr:`pts` MUST be scaled into :obj:`[0, 2^depth]`.
  '''

  # pts: (x, y, z, id)
  key = ocnn.octree.xyz2key(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], depth)
  idx = octree.search_key(key, depth, nempty)
  valid = idx > -1   # valid indices

  size = (pts.shape[0], data.shape[1])
  out = torch.zeros(size, device=data.device, dtype=data.dtype)
  out[valid] = data[idx[valid]]
  return out


def octree_trilinear_pts(data: torch.Tensor, octree: Octree, depth: int,
                         pts: torch.Tensor, nempty: bool = False):
  ''' Linear interpolatation with input points.

  Refer to :func:`octree_nearest_pts` for the meaning of the arguments.
  '''

  device = data.device
  grid = torch.Tensor(
      [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
       [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], device=device)

  # 1. Neighborhood searching
  xyzf = pts[:, :3] - 0.5   # the value is defined on the center of each voxel
  xyzi = xyzf.floor()       # the integer part  (N, 3)
  frac = xyzf - xyzi        # the fraction part (N, 3)

  xyzn = (xyzi.unsqueeze(1) + grid).view(-1, 3)
  batch = pts[:, 3].unsqueeze(1).repeat(1, 8).view(-1)
  key = ocnn.octree.xyz2key(xyzn[:, 0], xyzn[:, 1], xyzn[:, 2], batch, depth)
  idx = octree.search_key(key, depth, nempty)
  valid = idx > -1  # valid indices
  idx = idx[valid]

  # 2. Build the sparse matrix
  npt = pts.shape[0]
  ids = torch.arange(npt, device=idx.device)
  ids = ids.unsqueeze(1).repeat(1, 8).view(-1)
  ids = ids[valid]
  indices = torch.stack([ids, idx], dim=0).long()

  frac = (1 - grid) - frac.unsqueeze(dim=1)  # (8, 3) - (N, 1, 3) -> (N, 8, 3)
  weight = frac.prod(dim=2).abs().view(-1)   # (8*N,)
  weight = weight[valid]

  h = data.shape[0]
  mat = torch.sparse_coo_tensor(indices, weight, [npt, h], device=device)

  # 3. Interpolatation
  output = torch.sparse.mm(mat, data)
  ones = torch.ones(h, 1, dtype=data.dtype, device=device)
  norm = torch.sparse.mm(mat, ones)
  output = torch.div(output, norm + 1e-12)
  return output


class OctreeUpsample(torch.nn.Module):
  r''' Upsamples the octree node features with the nearest-neighbor interpolation.
  '''

  def __init__(self, nempty):
    super().__init__()
    self.nempty = nempty

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    if not self.nempty:
      data = ocnn.nn.octree_depad(data, octree, depth)
    out = out.unsqueeze(1).repeat(1, 8, 1).flatten(end_dim=1)
    if self.nempty:
      out = ocnn.nn.octree_depad(out, octree, depth+1)  # !!! depth+1
    return out
