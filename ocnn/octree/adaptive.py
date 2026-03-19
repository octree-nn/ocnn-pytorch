import torch
from ocnn.nn import octree_max_pool
from ocnn.octree import Octree
from ocnn.utils import range_grid, trilinear_interp_weights


def calculate_sdf_error(octree: Octree):
  rng = range_grid(0, 2, device=octree.device) / 2.0
  weights = trilinear_interp_weights(rng[:, 0], rng[:, 1], rng[:, 2])
  corners = torch.tensor([0, 2, 6, 8, 18, 20, 24, 26], device=octree.device)

  # calcuate the interpolation error for each node at each depth
  errors = {}
  for d in range(octree.full_depth + 1, octree.depth+1):
    fields_d = octree.fields[d]
    if fields_d.dtype == torch.int16:  # quantized fields
      fields_d = fields_d.float() / octree.field_scale  # int16 -> float
    fields_c = fields_d[:, corners]    # (N, 8)
    interp_d = fields_c @ weights.t()  # (N, 27)
    errors[d] = (fields_d - interp_d).abs().max(dim=1)[0]  # (N)

  # propagate the error from bottom to top
  for d in range(octree.depth, octree.full_depth + 1, -1):
    error_d = octree_max_pool(errors[d].unsqueeze(1), octree, d - 1, nempty=True)
    errors[d - 1] = torch.max(errors[d - 1], error_d.squeeze(1))
  return errors
