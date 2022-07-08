import torch
import torch.sparse
from typing import List

import ocnn
from ocnn.octree import Octree


def octree_nearest_pts(data: torch.Tensor, octree: Octree, depth: int,
                       pts: torch.Tensor, nempty: bool = False,
                       bound_check: bool = False):
  ''' The nearest-neighbor interpolatation with input points.

  Args:
    data (torch.Tensor): The input data.
    octree (Octree): The octree to interpolate.
    depth (int): The depth of the data.
    pts (torch.Tensor): The coordinates of the points with shape :obj:`(N, 4)`,
        i.e. :obj:`N x (x, y, z, batch)`.
    nempty (bool): If true, the :attr:`data` only contains features of non-empty 
        octree nodes
    bound_check (bool): If true, check whether the point is in :obj:`[0, 2^depth)`.

  .. note::
    The :attr:`pts` MUST be scaled into :obj:`[0, 2^depth)`.
  '''

  # pts: (x, y, z, id)
  key = ocnn.octree.xyz2key(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], depth)
  idx = octree.search_key(key, depth, nempty)
  valid = idx > -1   # valid indices
  if bound_check:
    bound = torch.logical_and(pts[:, :3] >= 0, pts[:, :3] < 2**depth).all(1)
    valid = torch.logical_and(valid, bound)

  size = (pts.shape[0], data.shape[1])
  out = torch.zeros(size, device=data.device, dtype=data.dtype)
  out[valid] = data[idx[valid]]
  return out


def octree_linear_pts(data: torch.Tensor, octree: Octree, depth: int,
                      pts: torch.Tensor, nempty: bool = False,
                      bound_check: bool = False):
  ''' Linear interpolatation with input points.

  Refer to :func:`octree_nearest_pts` for the meaning of the arguments.
  '''

  device = data.device
  grid = torch.tensor(
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
  if bound_check:
    bound = torch.logical_and(xyzn >= 0, xyzn < 2**depth).all(1)
    valid = torch.logical_and(valid, bound)
  idx = idx[valid]

  # 2. Build the sparse matrix
  npt = pts.shape[0]
  ids = torch.arange(npt, device=idx.device)
  ids = ids.unsqueeze(1).repeat(1, 8).view(-1)
  ids = ids[valid]
  indices = torch.stack([ids, idx], dim=0).long()

  frac = (1.0 - grid) - frac.unsqueeze(dim=1)  # (8, 3) - (N, 1, 3) -> (N, 8, 3)
  weight = frac.prod(dim=2).abs().view(-1)     # (8*N,)
  weight = weight[valid]

  h = data.shape[0]
  mat = torch.sparse_coo_tensor(indices, weight, [npt, h], device=device)

  # 3. Interpolatation
  output = torch.sparse.mm(mat, data)
  ones = torch.ones(h, 1, dtype=data.dtype, device=device)
  norm = torch.sparse.mm(mat, ones)
  output = torch.div(output, norm + 1e-12)
  return output


class OctreeInterp(torch.nn.Module):
  r''' Interpolates the points with an octree feature.

  Refer to :func:`octree_nearest_pts` for a description of arguments.
  '''

  def __init__(self, method: str = 'linear', nempty: bool = False,
               bound_check: bool = False, rescale_pts: bool = True):
    super().__init__()
    self.method = method
    self.nempty = nempty
    self.bound_check = bound_check
    self.rescale_pts = rescale_pts
    self.func = octree_linear_pts if method == 'linear' else octree_nearest_pts

  def forward(self, data: torch.Tensor, octree: Octree, depth: int,
              pts: torch.Tensor):
    r''''''

    # rescale points from [-1, 1] to [0, 2^depth]
    if self.rescale_pts:
      scale = 2 ** (depth - 1)
      pts[:, :3] = (pts[:, :3] + 1.0) * scale

    return self.func(data, octree, depth, pts, self.nempty, self.bound_check)

  def extra_repr(self) -> str:
    r''' Sets the extra representation of the module.
    '''

    return ('method={}, nempty={}, bound_check={}, rescale_pts={}').format(
            self.method, self.nempty, self.bound_check, self.rescale_pts)  # noqa


def octree_nearest_upsample(data: torch.Tensor, octree: Octree, depth: int,
                            nempty: bool = False):
  r''' Upsamples the octree node features from :attr:`depth` to :attr:`(depth+1)`
  with the nearest-neighbor interpolation.

  Args:
    data (torch.Tensor): The input data.
    octree (Octree): The octree to interpolate.
    depth (int): The depth of the data.
    nempty (bool): If true, the :attr:`data` only contains features of non-empty 
        octree nodes
  '''

  out = data
  if not nempty:
    out = ocnn.nn.octree_depad(out, octree, depth)
  out = out.unsqueeze(1).repeat(1, 8, 1).flatten(end_dim=1)
  if nempty:
    out = ocnn.nn.octree_depad(out, octree, depth+1)  # !!! depth+1
  return out


def octree_linear_upsample(data: torch.Tensor, octree: Octree, depth: int,
                           nempty: bool = False):
  r''' Upsamples the octree node features from :attr:`depth` to :attr:`(depth+1)`
  with the linear interpolation.

  Please refer to :func:`octree_upsample_nearest` for the arguments.
  '''

  xyzb = octree.xyzb(depth+1, nempty)
  pts = torch.stack(xyzb, dim=1)
  pts[:, :3] = (pts[:, :3] + 0.5) * 0.5
  out = octree_linear_pts(data, octree, depth, pts, nempty)
  return out


class OctreeUpsample(torch.nn.Module):
  r''' Upsamples the octree node features.

  Refer to :func:`octree_upsample` for details.
  '''

  def __init__(self, method: str = 'linear', nempty: bool = False):
    super().__init__()
    self.method = method
    self.nempty = nempty
    fn = {'linear': octree_linear_upsample, 'nearest': octree_nearest_upsample}
    self.func = fn[method]

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    return self.func(data, octree, depth, self.nempty)

  def extra_repr(self) -> str:
    r''' Sets the extra representation of the module.
    '''

    return ('method={}, nempty={}').format(self.method, self.nempty)
