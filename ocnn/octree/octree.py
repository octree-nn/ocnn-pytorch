# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn.functional as F
from typing import Union, List

import ocnn
from ocnn.octree.points import Points
from ocnn.octree.shuffled_key import xyz2key, key2xyz
from ocnn.utils import range_grid, scatter_add, cumsum, trunc_div


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
    The octree data structure requires that if an octree node has children nodes,
    the number of children nodes is exactly 8, in which some of the nodes are
    empty and some nodes are non-empty. The properties of an octree, including
    :obj:`keys`, :obj:`children` and :obj:`neighs`, contain both non-empty and
    empty nodes, and other properties, including :obj:`features`, :obj:`normals`
    and :obj:`points`, contain only non-empty nodes.

  .. note::
    The point cloud must be in range :obj:`[-1, 1]`.
  '''

  def __init__(self, depth: int, full_depth: int = 2, batch_size: int = 1,
               device: Union[torch.device, str] = 'cpu', **kwargs):
    super().__init__()
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

    # octree node numbers in each octree layers.
    # TODO: decide whether to settle them to 'gpu' or not?
    self.nnum = torch.zeros(num, dtype=torch.int32)
    self.nnum_nempty = torch.zeros(num, dtype=torch.int32)

    # the following properties are valid after `merge_octrees`.
    # TODO: make them valid after `octree_grow`, `octree_split` and `build_octree`
    batch_size = self.batch_size
    self.batch_nnum = torch.zeros(num, batch_size, dtype=torch.int32)
    self.batch_nnum_nempty = torch.zeros(num, batch_size, dtype=torch.int32)

    # construct the look up tables for neighborhood searching
    device = self.device
    center_grid = range_grid(2, 3, device)    # (8, 3)
    displacement = range_grid(-1, 1, device)  # (27, 3)
    neigh_grid = center_grid.unsqueeze(1) + displacement  # (8, 27, 3)
    parent_grid = trunc_div(neigh_grid, 2)
    child_grid = neigh_grid % 2
    self.lut_parent = torch.sum(
        parent_grid * torch.tensor([9, 3, 1], device=device), dim=2)
    self.lut_child = torch.sum(
        child_grid * torch.tensor([4, 2, 1], device=device), dim=2)

    # lookup tables for different kernel sizes
    self.lut_kernel = {
        '222': torch.tensor([13, 14, 16, 17, 22, 23, 25, 26], device=device),
        '311': torch.tensor([4, 13, 22], device=device),
        '131': torch.tensor([10, 13, 16], device=device),
        '113': torch.tensor([12, 13, 14], device=device),
        '331': torch.tensor([1, 4, 7, 10, 13, 16, 19, 22, 25], device=device),
        '313': torch.tensor([3, 4, 5, 12, 13, 14, 21, 22, 23], device=device),
        '133': torch.tensor([9, 10, 11, 12, 13, 14, 15, 16, 17], device=device),
    }

  def key(self, depth: int, nempty: bool = False):
    r''' Returns the shuffled key of each octree node.

    Args:
      depth (int): The depth of the octree.
      nempty (bool): If True, returns the results of non-empty octree nodes.
    '''

    key = self.keys[depth]
    if nempty:
      mask = self.nempty_mask(depth)
      key = key[mask]
    return key

  def xyzb(self, depth: int, nempty: bool = False):
    r''' Returns the xyz coordinates and the batch indices of each octree node.

    Args:
      depth (int): The depth of the octree.
      nempty (bool): If True, returns the results of non-empty octree nodes.
    '''

    key = self.key(depth, nempty)
    return key2xyz(key, depth)

  def batch_id(self, depth: int, nempty: bool = False):
    r''' Returns the batch indices of each octree node.

    Args:
      depth (int): The depth of the octree.
      nempty (bool): If True, returns the results of non-empty octree nodes.
    '''

    batch_id = self.keys[depth] >> 48
    if nempty:
      mask = self.nempty_mask(depth)
      batch_id = batch_id[mask]
    return batch_id

  def nempty_mask(self, depth: int):
    r''' Returns a binary mask which indicates whether the cooreponding octree
    node is empty or not.

    Args:
      depth (int): The depth of the octree.
    '''

    return self.children[depth] >= 0

  def build_octree(self, point_cloud: Points):
    r''' Builds an octree from a point cloud.

    Args:
      point_cloud (Points): The input point cloud.

    .. note::
      Currently, the batch size of the point cloud must be 1.
    '''

    self.device = point_cloud.device
    assert point_cloud.batch_size == self.batch_size, 'Inconsistent batch_size'

    # normalize points from [-1, 1] to [0, 2^depth]. #[L:Scale]
    scale = 2 ** (self.depth - 1)
    points = (point_cloud.points + 1.0) * scale

    # get the shuffled key and sort
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    b = None if self.batch_size == 1 else point_cloud.batch_id.view(-1)
    key = xyz2key(x, y, z, b, self.depth)
    node_key, idx, counts = torch.unique(
        key, sorted=True, return_inverse=True, return_counts=True)

    # layer 0 to full_layer: the octree is full in these layers
    for d in range(self.full_depth+1):
      self.octree_grow_full(d, update_neigh=False)

    # layer depth_ to full_layer_
    for d in range(self.depth, self.full_depth, -1):
      # compute parent key, i.e. keys of layer (d -1)
      pkey = node_key >> 3
      pkey, pidx, pcounts = torch.unique_consecutive(
          pkey, return_inverse=True, return_counts=True)

      # augmented key
      key = (pkey.unsqueeze(-1) << 3) + torch.arange(8, device=self.device)
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
      # Use `pkey >> 45` instead of `pkey >> 48` in L199 since pkey is already
      # shifted to the right by 3 bits in L177
      node_key = pkey if self.batch_size == 1 else  \
          ((pkey >> 45) << 48) | (pkey & ((1 << 45) - 1))

    # set the children for the layer full_layer,
    # now the node_keys are the key for full_layer
    d = self.full_depth
    children = -torch.ones_like(self.children[d])
    nempty_idx = node_key if self.batch_size == 1 else  \
        ((node_key >> 48) << (3 * d)) | (node_key & ((1 << 48) - 1))
    children[nempty_idx] = torch.arange(
        node_key.numel(), dtype=torch.int32, device=self.device)
    self.children[d] = children
    self.nnum_nempty[d] = node_key.numel()

    # average the signal for the last octree layer
    d = self.depth
    points = scatter_add(points, idx, dim=0)  # points is rescaled in [L:Scale]
    self.points[d] = points / counts.unsqueeze(1)
    if point_cloud.normals is not None:
      normals = scatter_add(point_cloud.normals, idx, dim=0)
      self.normals[d] = F.normalize(normals)
    if point_cloud.features is not None:
      features = scatter_add(point_cloud.features, idx, dim=0)
      self.features[d] = features / counts.unsqueeze(1)

    return idx

  def octree_grow_full(self, depth: int, update_neigh: bool = True):
    r''' Builds the full octree, which is essentially a dense volumetric grid.

    Args:
      depth (int): The depth of the octree.
      update_neigh (bool): If True, construct the neighborhood indices.
    '''

    # check
    assert depth <= self.full_depth, 'error'

    # node number
    num = 1 << (3 * depth)
    self.nnum[depth] = num * self.batch_size
    self.nnum_nempty[depth] = num * self.batch_size

    # update key
    key = torch.arange(num, dtype=torch.long, device=self.device)
    bs = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
    key = key.unsqueeze(0) | (bs.unsqueeze(1) << 48)
    self.keys[depth] = key.view(-1)

    # update children
    self.children[depth] = torch.arange(
        num * self.batch_size, dtype=torch.int32, device=self.device)

    # update neigh if needed
    if update_neigh:
      self.construct_neigh(depth)

  def octree_split(self, split: torch.Tensor, depth: int):
    r''' Sets whether the octree nodes in :attr:`depth` are splitted or not.

    Args:
      split (torch.Tensor): The input tensor with its element indicating status
          of each octree node: 0 - empty, 1 - non-empty or splitted.
      depth (int): The depth of current octree.
    '''

    # split -> children
    empty = split == 0
    sum = cumsum(split, dim=0, exclusive=True)
    children, nnum_nempty = torch.split(sum, [split.shape[0], 1])
    children[empty] = -1

    # boundary case, make sure that at least one octree node is splitted
    if nnum_nempty == 0:
      nnum_nempty = 1
      children[0] = 0

    # update octree
    self.children[depth] = children
    self.nnum_nempty[depth] = nnum_nempty

  def octree_grow(self, depth: int, update_neigh: bool = True):
    r''' Grows the octree and updates the relevant properties. And in most
    cases, call :func:`Octree.octree_split` to update the splitting status of
    the octree before this function.

    Args:
      depth (int): The depth of the octree.
      update_neigh (bool): If True, construct the neighborhood indices.
    '''

    # node number
    nnum = self.nnum_nempty[depth-1] * 8
    self.nnum[depth] = nnum
    self.nnum_nempty[depth] = nnum

    # update keys
    key = self.key(depth-1, nempty=True)
    batch_id = (key >> 48) << 48
    key = (key & ((1 << 48) - 1)) << 3
    key = key | batch_id
    key = key.unsqueeze(1) + torch.arange(8, device=key.device)
    self.keys[depth] = key.view(-1)

    # update children
    self.children[depth] = torch.arange(
        nnum, dtype=torch.int32, device=self.device)

    # update neighs
    if update_neigh:
      self.construct_neigh(depth)

  def construct_neigh(self, depth: int):
    r''' Constructs the :obj:`3x3x3` neighbors for each octree node.

    Args:
      depth (int): The octree depth with a value larger than 0 (:obj:`>0`).
    '''

    if depth <= self.full_depth:
      device = self.device
      nnum = 1 << (3 * depth)
      key = torch.arange(nnum, dtype=torch.long, device=device)
      x, y, z, _ = key2xyz(key, depth)
      xyz = torch.stack([x, y, z], dim=-1)  # (N,  3)
      grid = range_grid(-1, 1, device)   # (27, 3)
      xyz = xyz.unsqueeze(1) + grid         # (N, 27, 3)
      xyz = xyz.view(-1, 3)                 # (N*27, 3)
      neigh = xyz2key(xyz[:, 0], xyz[:, 1], xyz[:, 2], depth=depth)

      bs = torch.arange(self.batch_size, dtype=torch.int32, device=device)
      neigh = neigh + bs.unsqueeze(1) * nnum  # (N*27,) + (B, 1) -> (B, N*27)

      bound = 1 << depth
      invalid = torch.logical_or((xyz < 0).any(1), (xyz >= bound).any(1))
      neigh[:, invalid] = -1
      self.neighs[depth] = neigh.view(-1, 27)  # (B*N, 27)

    else:
      child_p = self.children[depth-1]
      nempty = child_p >= 0
      neigh_p = self.neighs[depth-1][nempty]   # (N, 27)
      neigh_p = neigh_p[:, self.lut_parent]    # (N, 8, 27)
      child_p = child_p[neigh_p]               # (N, 8, 27)
      invalid = torch.logical_or(child_p < 0, neigh_p < 0)   # (N, 8, 27)
      neigh = child_p * 8 + self.lut_child
      neigh[invalid] = -1
      self.neighs[depth] = neigh.view(-1, 27)

  def construct_all_neigh(self):
    r''' A convenient handler for constructing all neighbors.
    '''

    for depth in range(1, self.depth+1):
      self.construct_neigh(depth)

  def search_xyzb(self, query: torch.Tensor, depth: int, nempty: bool = False):
    r''' Searches the octree nodes given the query points.

    Args:
      query (torch.Tensor): The coordinates of query points with shape
          :obj:`(N, 4)`. The first 3 channels of the coordinates are :obj:`x`,
          :obj:`y`, and :obj:`z`, and the last channel is the batch index. Note
          that the coordinates must be in range :obj:`[0, 2^depth)`.
      depth (int): The depth of the octree layer. nemtpy (bool): If true, only
          searches the non-empty octree nodes.
    '''

    key = xyz2key(query[:, 0], query[:, 1], query[:, 2], query[:, 3], depth)
    idx = self.search_key(key, depth, nempty)
    return idx

  def search_key(self, query: torch.Tensor, depth: int, nempty: bool = False):
    r''' Searches the octree nodes given the query points.

    Args:
      query (torch.Tensor): The keys of query points with shape :obj:`(N,)`,
          which are computed from the coordinates of query points.
      depth (int): The depth of the octree layer. nemtpy (bool): If true, only
          searches the non-empty octree nodes.
    '''

    key = self.key(depth, nempty)
    # `torch.bucketize` is similar to `torch.searchsorted`.
    # I choose `torch.bucketize` here because it has fewer dimension checks,
    # resulting in slightly better performance according to the docs of
    # pytorch-1.9.1, since `key` is always 1-D sorted sequence.
    idx = torch.bucketize(query, key)

    valid = idx < key.shape[0]      # valid if NOT out-of-bound
    found = key[idx[valid]] == query[valid]
    valid[valid.clone()] = found    # valid if found
    idx[valid.logical_not()] = -1   # set to -1 if invalid
    return idx

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
      # clone neigh to avoid self.neigh[depth] being modified
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

  def get_input_feature(self, feature: str, nempty: bool = False):
    r''' Returns the initial input feature stored in octree.

    Args:
      feature (str): A string used to indicate which features to extract from
          the input octree. If the character :obj:`N` is in :attr:`feature`, the
          normal signal is extracted (3 channels). Similarly, if :obj:`D` is in
          :attr:`feature`, the local displacement is extracted (1 channels). If
          :obj:`L` is in :attr:`feature`, the local coordinates of the averaged
          points in each octree node is extracted (3 channels). If :attr:`P` is
          in :attr:`feature`, the global coordinates are extracted (3 channels).
          If :attr:`F` is in :attr:`feature`, other features (like colors) are
          extracted (k channels).
      nempty (bool): If false, gets the features of all octree nodes.
    '''

    features = list()
    depth = self.depth
    feature = feature.upper()
    if 'N' in feature:
      features.append(self.normals[depth])

    if 'L' in feature or 'D' in feature:
      local_points = self.points[depth].frac() - 0.5

    if 'D' in feature:
      dis = torch.sum(local_points * self.normals[depth], dim=1, keepdim=True)
      features.append(dis)

    if 'L' in feature:
      features.append(local_points)

    if 'P' in feature:
      scale = 2 ** (1 - depth)   # normalize [0, 2^depth] -> [-1, 1]
      global_points = self.points[depth] * scale - 1.0
      features.append(global_points)

    if 'F' in feature:
      features.append(self.features[depth])

    out = torch.cat(features, dim=1)
    if not nempty:
      out = ocnn.nn.octree_pad(out, self, depth)
    return out

  def to_points(self, rescale: bool = True):
    r''' Converts averaged points in the octree to a point cloud.

    Args:
      rescale (bool): rescale the xyz coordinates to [-1, 1] if True.
    '''

    depth = self.depth
    batch_size = self.batch_size

    # by default, use the average points generated when building the octree
    # from the input point cloud
    xyz = self.points[depth]
    batch_id = self.batch_id(depth, nempty=True)

    # xyz is None when the octree is predicted by a neural network
    if xyz is None:
      x, y, z, batch_id = self.xyzb(depth, nempty=True)
      xyz = torch.stack([x, y, z], dim=1) + 0.5

    # normalize xyz to [-1, 1] since the average points are in range [0, 2^d]
    if rescale:
      scale = 2 ** (1 - depth)
      xyz = xyz * scale - 1.0

    # construct Points
    out = Points(xyz, self.normals[depth], self.features[depth],
                 batch_id=batch_id.unsqueeze(1), batch_size=batch_size)
    return out

  def to(self, device: Union[torch.device, str], non_blocking: bool = False):
    r''' Moves the octree to a specified device.

    Args:
      device (torch.device or str): The destination device.
      non_blocking (bool): If True and the source is in pinned memory, the copy
          will be asynchronous with respect to the host. Otherwise, the argument
          has no effect. Default: False.
    '''

    if isinstance(device, str):
      device = torch.device(device)

    #  If on the save device, directly retrun self
    if self.device == device:
      return self

    def list_to_device(prop):
      return [p.to(device, non_blocking=non_blocking)
              if isinstance(p, torch.Tensor) else None for p in prop]

    # Construct a new Octree on the specified device
    octree = Octree(self.depth, self.full_depth, self.batch_size, device)
    octree.keys = list_to_device(self.keys)
    octree.children = list_to_device(self.children)
    octree.neighs = list_to_device(self.neighs)
    octree.features = list_to_device(self.features)
    octree.normals = list_to_device(self.normals)
    octree.points = list_to_device(self.points)
    octree.nnum = self.nnum.clone()  # TODO: whether to move nnum to the self.device?
    octree.nnum_nempty = self.nnum_nempty.clone()
    octree.batch_nnum = self.batch_nnum.clone()
    octree.batch_nnum_nempty = self.batch_nnum_nempty.clone()
    return octree

  def cuda(self, non_blocking: bool = False):
    r''' Moves the octree to the GPU. '''

    return self.to('cuda', non_blocking)

  def cpu(self):
    r''' Moves the octree to the CPU. '''

    return self.to('cpu')


def merge_octrees(octrees: List['Octree']):
  r''' Merges a list of octrees into one batch.

  Args:
    octrees (List[Octree]): A list of octrees to merge.

  Returns:
    Octree: The merged octree.
  '''

  # init and check
  octree = Octree(depth=octrees[0].depth, full_depth=octrees[0].full_depth,
                  batch_size=len(octrees), device=octrees[0].device)
  for i in range(1, octree.batch_size):
    condition = (octrees[i].depth == octree.depth and
                 octrees[i].full_depth == octree.full_depth and
                 octrees[i].device == octree.device)
    assert condition, 'The check of merge_octrees failed'

  # node num
  batch_nnum = torch.stack(
      [octrees[i].nnum for i in range(octree.batch_size)], dim=1)
  batch_nnum_nempty = torch.stack(
      [octrees[i].nnum_nempty for i in range(octree.batch_size)], dim=1)
  octree.nnum = torch.sum(batch_nnum, dim=1)
  octree.nnum_nempty = torch.sum(batch_nnum_nempty, dim=1)
  octree.batch_nnum = batch_nnum
  octree.batch_nnum_nempty = batch_nnum_nempty
  nnum_cum = cumsum(batch_nnum_nempty, dim=1, exclusive=True)

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
      child = octrees[i].children[d].clone()  # !! `clone` is used here to avoid
      mask = child >= 0                       # !! modifying the original octrees
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


def init_octree(depth: int, full_depth: int = 2, batch_size: int = 1,
                device: Union[torch.device, str] = 'cpu'):
  r'''
  Initializes an octree to :attr:`full_depth`.

  Args:
    depth (int): The depth of the octree.
    full_depth (int): The octree layers with a depth small than
        :attr:`full_depth` are forced to be full.
    batch_size (int, optional): The batch size.
    device (torch.device or str): The device to use for computation.

  Returns:
    Octree: The initialized Octree object.
  '''

  octree = Octree(depth, full_depth, batch_size, device)
  for d in range(full_depth+1):
    octree.octree_grow_full(depth=d)
  return octree
