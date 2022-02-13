import torch
import torch.nn.functional as F
from typing import Optional, Union

from .points import Points
from .shuffled_key import xyz2key
from .scatter import scatter_add


class Octree:
  r''' Builds an octree from an input point cloud.

  Args:
    depth (int): The octree depth.
    full_depth (int): The octree layers with a depth small than
        :attr:`full_depth` are forced to be full.

  .. note::
    The point cloud must be in range :obj:`[-1, 1]`.
  '''

  def __init__(self, depth: int, full_depth: int = 2, **kwargs):
    self.depth = depth
    self.full_depth = full_depth
    depth1 = self.depth + 1

    self.keys = [None] * depth1
    self.children = [None] * depth1
    self.neighs = [None] * depth1
    self.features = [None] * depth1
    self.normals = [None] * depth1
    self.points = [None] * depth1

    self.nnum = torch.zeros(depth1, dtype=torch.int32)
    self.nnum_cum = torch.zeros(depth1, dtype=torch.int32)
    self.nnum_nempty = torch.zeros(depth1, dtype=torch.int32)
    self.batch_size = 1
    self.device = torch.device('cpu')

  def build_octree(self, point_cloud: Points):
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
    normals = scatter_add(point_cloud.normals, idx, dim=0)
    self.normals[self.depth] = F.normalize(normals)
    features = scatter_add(point_cloud.features, idx, dim=0)
    self.features[self.depth] = features / counts.unsqueeze(1)
    points = scatter_add(point_cloud.points, idx, dim=0)
    self.points[self.depth] = points / counts.unsqueeze(1)
