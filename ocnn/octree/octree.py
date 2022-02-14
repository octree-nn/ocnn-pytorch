import torch
import torch.nn.functional as F
from typing import Optional, Union, List

from .points import Points
from .shuffled_key import xyz2key
from .scatter import scatter_add


class Octree:
  r''' Builds an octree from an input point cloud.

  Args:
    depth (int): The octree depth.
    full_depth (int): The octree layers with a depth small than
        :attr:`full_depth` are forced to be full.
    device (str): Choose from :obj:`cpu` and :obj:`gpu`. (default: :obj:`cpu`)

  .. note::
    The point cloud must be in range :obj:`[-1, 1]`.
  '''

  def __init__(self, depth: int, full_depth: int = 2, device: str = 'cpu', **kwargs):
    self.depth = depth
    self.full_depth = full_depth

    self.reset()
    self.batch_size = 1
    self.device = device

  def reset(self):
    num = self.depth + 1
    self.keys = [None] * num
    self.children = [None] * num
    self.neighs = [None] * num
    self.features = [None] * num
    self.normals = [None] * num
    self.points = [None] * num

    self.nnum = torch.zeros(num, dtype=torch.int32)
    # self.nnum_cum = torch.zeros(num, dtype=torch.int32)
    self.nnum_nempty = torch.zeros(num, dtype=torch.int32)

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
    points = scatter_add(point_cloud.points, idx, dim=0)
    self.points[self.depth] = points / counts.unsqueeze(1)
    if point_cloud.normals is not None:
      normals = scatter_add(point_cloud.normals, idx, dim=0)
      self.normals[self.depth] = F.normalize(normals)
    if point_cloud.features is not None:
      features = scatter_add(point_cloud.features, idx, dim=0)
      self.features[self.depth] = features / counts.unsqueeze(1)

  def merge_octrees(self, octrees: List['Octree']):
    r''' Merges a list of octrees into one batch. '''

    # init and check
    self.batch_size = len(octrees)
    self.depth = octrees[0].depth
    self.full_depth = octrees[0].full_depth
    self.device = octrees[0].device
    self.reset()
    for i in range(1, self.batch_size):
      condition = (octrees[i].depth == self.depth and
                   octrees[i].full_depth == self.full_depth and
                   octrees[i].device == self.device)
      torch._assert(condition, 'The check of merge_octrees failed')

    # node num
    nnum = torch.stack(
        [octrees[i].nnum for i in range(self.batch_size)], dim=1)
    nnum_nempty = torch.stack(
        [octrees[i].nnum_nempty for i in range(self.batch_size)], dim=1)
    self.nnum = torch.sum(nnum, dim=1)
    self.nnum_nempty = torch.sum(nnum_nempty, dim=1)
    nnum_cum = torch.cumsum(nnum_nempty, dim=1) - nnum_nempty[:, :1]

    # merge octre properties
    for d in range(self.depth+1):
      # key
      keys = [None] * self.batch_size
      for i in range(self.batch_size):
        key = octrees[i].keys[d] & ((1 << 48) - 1)  # clear the highest bits
        keys[i] = key | d << 48
      self.keys[d] = torch.cat(keys, dim=0)

      # children
      children = [None] * self.batch_size
      for i in range(self.batch_size):
        child = octrees[i].children[d]
        mask = child >= 0
        child[mask] = child[mask] + nnum_cum[d, i]
        children[i] = child
      self.children[d] = torch.cat(children, dim=0)

      # features
      if octrees[0].features[d] is not None and d == self.depth:
        features = [octrees[i].features[d] for i in range(self.batch_size)]
        self.features[d] = torch.cat(features, dim=0)

      # normals
      if octrees[0].normals[d] is not None and d == self.depth:
        normals = [octrees[i].normals[d] for i in range(self.batch_size)]
        self.normals[d] = torch.cat(normals, dim=0)

      # points
      if octrees[0].points[d] is not None and d == self.depth:
        points = [octrees[i].points[d] for i in range(self.batch_size)]
        self.points[d] = torch.cat(points, dim=0)

  def compute_neigh(self, depth: int):

    # init
    if depth < 1: return
    nnum = self.nnum[depth]
    self.neighs[depth] = - torch.ones(
        nnum // 8, 64, dtype=torch.int32, device=self.device)

    # construct neigh when depth == 1
    if depth == 1:
      # x = [0,0,0,0,1,1,1,1], y = [0,0,1,1,0,0,1,1], z = [0,1,0,1,0,1,0,1]
      # addr = ((x+1) << 4) | ((y+1) << 2) | (z+1)
      addr = torch.tensor([21, 22, 25, 26, 37, 38, 41, 42], device=self.device)
      neigh = - torch.ones(64, dtype=torch.int64, device=self.device)
      neigh[addr] = torch.arange(8, dtype=torch.int64, device=self.device)
      self.neighs[depth] = neigh
      return
    
    # construct other neighs
    neigh_parent = self.neighs[depth-1]
    
