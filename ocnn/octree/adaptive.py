from ocnn.octree import Points, init_octree, Octree, xyz2key
import torch
import torch.nn.functional as F
import ocnn

def build_adaptive_octree_normal(
    pointcloud: Points, depth: int, full_depth: int,
    adaptive_depth: int, depth_list: list = None,
    th_dist: float = 0.3, th_norm: float = 1e-3,
):
  '''
  Build an adaptive octree from a point cloud.
  Args:
    pointcloud: The input point cloud.
    depth: The max depth of the octree.
    full_depth: The full depth of the octree.
    adaptive_depth: The depth at which the octree becomes adaptive.
    depth_list: The list of depths at which the octree is adaptive.
    th_dist: The threshold for the distance error.
    th_norm: The threshold for the normal error.
  
  Returns:
    The adaptive octree.
  '''
  points = pointcloud.points
  normals = pointcloud.normals
  device = pointcloud.device
  batch_size = pointcloud.batch_size
  octree = Octree(depth, full_depth, batch_size, device)

  if depth_list is None:
    depth_list = list(range(adaptive_depth, octree.depth + 1))

  for d in range(octree.full_depth + 1):
    octree.octree_grow_full(d)

  # --- Top-down Adaptive Construction ---
  # Iterate from the first adaptive layer to the maximum depth
  for d in range(octree.full_depth, octree.depth + 1):
    # If no points are left to process, create empty nodes and continue
    if points.shape[0] == 0:
      label = torch.zeros(octree.nnum[d], dtype=torch.int32, device=device)
      octree.octree_split(label, d)
      if d < octree.depth:
        octree.octree_grow(d + 1)
      octree.points[d] = torch.zeros(octree.nnum[d], 3, device=device)
      octree.normals[d] = torch.zeros(octree.nnum[d], 3, device=device)
      continue

    # 1. Assign points to the nodes at the current depth 'd'
    scale = 2 ** (d - 1)
    scaled_points = (points + 1.0) * scale
    x, y, z = scaled_points[:, 0], scaled_points[:, 1], scaled_points[:, 2]
    key = xyz2key(x, y, z, None, octree.depth)
    idx = torch.searchsorted(octree.keys[d], key)

    # 2. Aggregate points and normals for each node
    num_d = octree.nnum[d]
    counts_per_node = torch.zeros(num_d, dtype=torch.long, device=device)
    counts_per_node.scatter_reduce_(0, idx, torch.ones_like(idx, dtype=torch.long), reduce='sum', include_self=False)

    # Calculate average point and normal for each non-empty node
    expand_idx = idx.unsqueeze(1).expand(-1, 3)
    avg_points = torch.zeros(num_d, 3, device=device).scatter_reduce_(
      0, expand_idx, scaled_points, reduce='mean', include_self=False)
    avg_normals = torch.zeros(num_d, 3, device=device).scatter_reduce_(
      0, expand_idx, normals, reduce='mean', include_self=False)
    avg_normals = F.normalize(avg_normals, p=2, dim=1)

    # Save the computed average values into the node array for this level
    octree.points[d] = avg_points
    octree.normals[d] = avg_normals

    if d in depth_list and d >= adaptive_depth:
      # 3. Calculate geometric errors for each node
      # Gather the average properties corresponding to each point
      points_gathered = avg_points[idx]
      normals_gathered = avg_normals[idx]

      # -- Distance Error: Max distance from a point to the node's average plane
      vec = scaled_points - points_gathered
      dist = torch.abs((vec * normals_gathered).sum(dim=1)) / scale
      dist_err = torch.zeros(num_d, device=device).scatter_reduce_(
        0, idx, dist, reduce='max', include_self=False)

      # -- Normal Error: Average deviation of point normals from the node's average normal
      cos_similarity = (normals * normals_gathered).sum(dim=1).clamp_(-1, 1)
      n_err_points = 1.0 - cos_similarity
      avg_n_err = torch.zeros(num_d, device=device).scatter_reduce_(
        0, idx, n_err_points, reduce='mean', include_self=False)

      # 4. Make splitting decision based on errors
      # A node is pruned (not split) if its errors are below the thresholds
      prune_mask = (dist_err < th_dist) & (avg_n_err < th_norm)

      # Label=1 means SPLIT, Label=0 means PRUNE (do not split)
      label = (~prune_mask) & (counts_per_node > 0)
      
      # 5. Filter the point cloud for the next iteration
      # Only points in nodes that were split are processed in the next level
      mask_to_keep = (~prune_mask)[idx].bool()
      points = points[mask_to_keep]
      normals = normals[mask_to_keep]
    else:
      label = counts_per_node > 0

    # 6. Perform octree operations: split and grow for the next level
    octree.octree_split(label.long(), d)
    if d < octree.depth:
      octree.octree_grow(d + 1)

  return octree


def get_input_feature_multiscale(octree, feature="ND", nempty=False):
  '''
  Adaptive octree needs to get the multiscale input feature.
  '''
  out = dict()
  depth = octree.depth
  full_depth = octree.full_depth
  feature = feature.upper()
  for d in range(full_depth, depth + 1):
    features = []
    if octree.points[d] == None or octree.normals[d] == None:
      continue
    # Only take features of leaf nodes
    if 'N' in feature:
      features.append(octree.normals[d])

    if 'L' in feature or 'D' in feature:
      local_points = octree.points[d].frac() - 0.5

    if 'D' in feature:
      dis = torch.sum(local_points * octree.normals[d], dim=1, keepdim=True)
      features.append(dis)

    if 'L' in feature:
      features.append(local_points)

    if 'P' in feature:
      scale = 2 ** (1 - d)   # normalize [0, 2^depth] -> [-1, 1]
      global_points = octree.points[d] * scale - 1.0
      features.append(global_points)

    if 'F' in feature:
      features.append(octree.features[d])
    
    features = torch.cat(features, dim=1)
    if nempty:
      nempty_mask = octree.children[d] == -1 if d < depth \
        else torch.ones((octree.nnum[d],), dtype=torch.bool, device=octree.device)
      features = features[nempty_mask]
    out[d] = features
    
  return out
