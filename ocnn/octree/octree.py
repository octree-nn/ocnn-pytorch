import torch
import torch.nn.functional as F
from typing import Optional, Union, List

from .points import Points
from .shuffled_key import xyz2key, key2xyz
from .scatter import scatter_add


class Octree:
  r''' Builds an octree from an input point cloud.

  Args:
    depth (int): The octree depth.
    full_depth (int): The octree layers with a depth small than
        :attr:`full_depth` are forced to be full.
    batch_size (int): The octree batch size.
    device (torch.device or str): Choose from :obj:`cpu` and :obj:`gpu`.
        (default: :obj:`cpu`)

  .. note::
    The point cloud must be in range :obj:`[-1, 1]`.
  '''

  def __init__(self, depth: int, full_depth: int = 2, batch_size: int = 1,
               device: Union[torch.device, str] = 'cpu', **kwargs):
    self.depth = depth
    self.full_depth = full_depth
    self.batch_size = batch_size
    self.device = device

    self.reset()

  def reset(self):
    r''' Resets the Octree status and constructs several lookup tables.
    '''

    # octree features in each octree layers
    num = self.depth + 1
    self.keys = [None] * num
    self.children = [None] * num
    self.neighs = [None] * num
    self.features = [None] * num
    self.normals = [None] * num
    self.points = [None] * num

    # octree node numbers in each octree layers TODO: settle them to 'gpu'?
    self.nnum = torch.zeros(num, dtype=torch.int32)
    self.nnum_nempty = torch.zeros(num, dtype=torch.int32)
    # self.nnum_cum = torch.zeros(num, dtype=torch.int32)

    # construct the look up tables for neighborhood searching
    center_grid = self.meshgrid(2, 3)    # (8, 3)
    displacement = self.meshgrid(-1, 1)  # (27, 3)
    neigh_grid = center_grid.unsqueeze(1) + displacement  # (8, 27, 3)
    parent_grid = torch.div(neigh_grid, 2, rounding_mode='trunc')
    child_grid = neigh_grid % 2
    self.lut_parent = torch.sum(
        parent_grid * torch.tensor([9, 3, 1]), dim=2).to(self.device)
    self.lut_child = torch.sum(
        child_grid * torch.tensor([4, 2, 1]), dim=2).to(self.device)

    # lookup tables for different kernel sizes
    self.lut_kernel = {
        '222': torch.tensor([13, 14, 16, 17, 22, 23, 25, 26], device=self.device),
        '311': torch.tensor([4, 13, 22], device=self.device),
        '131': torch.tensor([10, 13, 16], device=self.device),
        '113': torch.tensor([12, 13, 14], device=self.device),
        '331': torch.tensor([1, 4, 7, 10, 13, 16, 19, 22, 25], device=self.device),
        '313': torch.tensor([3, 4, 5, 12, 13, 14, 21, 22, 23], device=self.device),
        '133': torch.tensor([9, 10, 11, 12, 13, 14, 15, 16, 17], device=self.device),
    }

  def build_octree(self, point_cloud: Points):
    r''' Builds an octree from a point cloud.

    Args:
      point_cloud (Points): The input point cloud.
    '''

    self.device = point_cloud.device

    # normalize points from [-1, 1] to [0, 2^depth]
    scale = 2 ** (self.depth - 1)
    points = (point_cloud.points + 1.0) * scale

    # get the shuffled key and sort
    key = xyz2key(points[:, 0], points[:, 1], points[:, 2], depth=self.depth)
    node_key, idx, counts = torch.unique(
        key, sorted=True, return_inverse=True, return_counts=True)

    # layer 0 to full_layer: the octree is full in these layers
    for d in range(self.full_depth+1):
      num = 1 << (3 * d)
      self.keys[d] = torch.arange(num, dtype=torch.long, device=self.device)
      self.children[d] = torch.arange(
          num, dtype=torch.int32, device=self.device)
      self.nnum[d] = num
      self.nnum_nempty[d] = num

    # layer depth_ to full_layer_
    for d in range(self.depth, self.full_depth, -1):
      # compute parent key, i.e. keys of layer (d -1)
      pkey = node_key >> 3
      pkey, pidx, pcounts = torch.unique_consecutive(
          pkey, return_inverse=True, return_counts=True)

      # augmented key
      key = pkey.unsqueeze(-1) * 8 + torch.arange(8, device=self.device)
      self.keys[d] = key.view(-1)
      self.nnum[d] = key.numel()
      self.nnum_nempty[d] = node_key.numel()

      # children
      addr = (pidx << 3) | (node_key % 8)
      children = -torch.ones(
          self.nnum[d].item(), dtype=torch.int32, device=self.device)
      children[addr] = torch.arange(
          self.nnum_nempty[d], dtype=torch.int32, device=self.device)
      self.children[d] = children

      # cache pkey for the next iteration
      node_key = pkey

    # set the children for the layer full_layer,
    # now the node_keys are the key for full_layer
    d = self.full_depth
    children = -torch.ones_like(self.children[d])
    children[node_key] = torch.arange(
        node_key.numel(), dtype=torch.int32, device=self.device)
    self.children[d] = children
    self.nnum_nempty[d] = node_key.numel()

    # average the signal for the last octree layer
    d = self.depth
    points = scatter_add(points, idx, dim=0)  # here points is rescaled in L84
    self.points[d] = points / counts.unsqueeze(1)
    if point_cloud.normals is not None:
      normals = scatter_add(point_cloud.normals, idx, dim=0)
      self.normals[d] = F.normalize(normals)
    if point_cloud.features is not None:
      features = scatter_add(point_cloud.features, idx, dim=0)
      self.features[d] = features / counts.unsqueeze(1)

    return idx

  def construct_neigh(self, depth: int):
    r''' Constructs the :obj:`3x3x3` neighbors for each octree node.

    Args:
      depth (int): The octree depth with a value larger than 0 (:obj:`>0`).
    '''

    if depth <= self.full_depth:
      nnum = 1 << (3 * depth)
      key = torch.arange(nnum, dtype=torch.long, device=self.device)
      x, y, z, _ = key2xyz(key, depth)
      xyz = torch.stack([x, y, z], dim=-1)  # (N,  3)
      grid = self.meshgrid(min=-1, max=1)   # (27, 3)
      xyz = xyz.unsqueeze(1) + grid         # (N, 27, 3)
      xyz = xyz.view(-1, 3)                 # (N*27, 3)
      neigh = xyz2key(xyz[:, 0], xyz[:, 1], xyz[:, 2], depth=depth)

      bs = torch.arange(self.batch_size, dtype=torch.int32, device=self.device)
      neigh = neigh + bs.unsqueeze(1) * nnum  # (N*27,) + (B, 1) -> (B, N*27)

      bound = 1 << depth
      invalid = torch.logical_or((xyz < 0).any(1), (xyz >= bound).any(1))
      neigh[:, invalid] = -1
      self.neighs[depth] = neigh.view(-1, 27)  # (B*N, 27)

    else:
      child_p = self.children[depth-1]
      mask = child_p >= 0
      neigh_p = self.neighs[depth-1][mask]   # (N, 27)
      neigh_p = neigh_p[:, self.lut_parent]  # (N, 8, 27)
      child_p = child_p[neigh_p]  # (N, 8, 27)
      invalid = child_p < 0       # (N, 8, 27)
      neigh = child_p * 8 + self.lut_child
      neigh[invalid] = -1
      self.neighs[depth] = neigh.view(-1, 27)

  def construct_all_neigh(self):
    r''' A convenient handler for constructing all neighbors.
    '''

    for depth in range(1, self.depth+1):
      self.construct_neigh(depth)

  def get_neigh(self, depth: int, kernel: str = '333', stride: int = 1,
                nempty: bool = False):
    r''' Returns the neighborhoods given the depth and a kernel shape.

    Args:
      depth (int): The octree depth with a value larger than 0 (:obj:`>0`).
      kernel (str): The kernel shape from :obj:`333`, :obj:`311`, :obj:`131`,
          :obj:`113`, :obj:`222`, :obj:`331`, :obj:`133`, and :obj:`313`.
      stride (int): The stride of neighborhoods (:obj:`1` or :obj:`2`). If the
          stride is :obj:`2`, always returns the neighborhood of the first
          siblings.
      nempty (bool): If True, only returns the neighborhoods of the non-empty
          octree nodes.
    '''

    if stride == 1:
      neigh = self.neighs[depth]
    elif stride == 2:
      # clone neigh to avoid self.neigh[depth] being modified (such as in L282)
      neigh = self.neighs[depth][::8].clone()
    else:
      raise ValueError('Unsupported stride {}'.format(stride))

    if nempty:
      child = self.children[depth]
      if stride == 1:
        nempty_node = child >= 0
        neigh = neigh[nempty_node]
      valid = neigh >= 0
      neigh[valid] = child[neigh[valid]].long()  # remap the index

    if kernel == '333':
      return neigh
    elif kernel in self.lut_kernel:
      lut = self.lut_kernel[kernel]
      return neigh[:, lut]
    else:
      raise ValueError('Unsupported kernel {}'.format(kernel))

  def get_input_feature(self):
    r''' Gets the initial input features.
    '''

    # normals
    features = list()
    depth = self.depth
    has_normal = self.normals[depth] is not None
    if has_normal:
      features.append(self.normals[depth])

    # local points
    points = self.points[depth].frac() - 0.5
    if has_normal:
      dis = torch.sum(points * self.normals[depth], dim=1, keepdim=True)
      features.append(dis)
    else:
      features.append(points)

    # features
    if self.features[depth] is not None:
      features.append(self.features[depth])

    return torch.cat(features, dim=0)

  def to_points(self):
    r''' Converts averaged points in the octree to a point cloud.
    '''

    d = self.depth
    scale = 2 ** (1-d)
    points = self.points[d] * scale - 1.0
    return Points(points, self.normals[d], self.features[d])

  def to(self, device: Union[torch.device, str]):
    r''' Moves the octree to a specified device.

    Args:
      device (torch.device or str): The destination device.
    '''

    def to_device(prop):
      for i in range(len(prop)):
        if isinstance(prop[i], torch.Tensor):
          prop[i] = prop[i].to(device)

    self.device = device
    self.keys = to_device(self.keys)
    self.children = to_device(self.children)
    self.neighs = to_device(self.neighs)
    self.features = to_device(self.features)
    self.normals = to_device(self.normals)
    self.points = to_device(self.points)

    self.lut_parent = self.lut_parent.to(device)
    self.lut_child = self.lut_child.to(device)
    for key, val in self.lut_kernel.items():
      self.lut_kernel[key] = val.to(device)

  def meshgrid(self, min, max):
    r''' Builds a mesh grid in :obj:`[min, max]` (:attr:`max` included).
    '''

    rng = torch.arange(min, max+1, dtype=torch.long, device=self.device)
    grid = torch.meshgrid(rng, rng, rng, indexing='ij')
    grid = torch.stack(grid, dim=-1).view(-1, 3)  # (27, 3)
    return grid


def merge_octrees(octrees: List['Octree']):
  r''' Merges a list of octrees into one batch.

  Args:
    octrees (List[Octree]): A list of octrees to merge.
  '''

  # init and check
  octree = Octree(depth=octrees[0].depth, full_depth=octrees[0].full_depth,
                  batch_size=len(octrees), device=octrees[0].device)
  for i in range(1, octree.batch_size):
    condition = (octrees[i].depth == octree.depth and
                 octrees[i].full_depth == octree.full_depth and
                 octrees[i].device == octree.device)
    torch._assert(condition, 'The check of merge_octrees failed')

  # node num
  nnum = torch.stack(
      [octrees[i].nnum for i in range(octree.batch_size)], dim=1)
  nnum_nempty = torch.stack(
      [octrees[i].nnum_nempty for i in range(octree.batch_size)], dim=1)
  octree.nnum = torch.sum(nnum, dim=1)
  octree.nnum_nempty = torch.sum(nnum_nempty, dim=1)

  nnum_cum = torch.cumsum(nnum_nempty, dim=1)
  pad = torch.zeros_like(octrees[0].nnum).unsqueeze(1)
  nnum_cum = torch.cat([pad, nnum_cum], dim=1)

  # merge octre properties
  for d in range(octree.depth+1):
    # key
    keys = [None] * octree.batch_size
    for i in range(octree.batch_size):
      key = octrees[i].keys[d] & ((1 << 48) - 1)  # clear the highest bits
      keys[i] = key | (i << 48)
    octree.keys[d] = torch.cat(keys, dim=0)

    # children
    children = [None] * octree.batch_size
    for i in range(octree.batch_size):
      child = octrees[i].children[d]
      mask = child >= 0
      child[mask] = child[mask] + nnum_cum[d, i]
      children[i] = child
    octree.children[d] = torch.cat(children, dim=0)

    # features
    if octrees[0].features[d] is not None and d == octree.depth:
      features = [octrees[i].features[d] for i in range(octree.batch_size)]
      octree.features[d] = torch.cat(features, dim=0)

    # normals
    if octrees[0].normals[d] is not None and d == octree.depth:
      normals = [octrees[i].normals[d] for i in range(octree.batch_size)]
      octree.normals[d] = torch.cat(normals, dim=0)

    # points
    if octrees[0].points[d] is not None and d == octree.depth:
      points = [octrees[i].points[d] for i in range(octree.batch_size)]
      octree.points[d] = torch.cat(points, dim=0)

  return octree
