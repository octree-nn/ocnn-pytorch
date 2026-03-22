import torch
from ocnn.nn import octree_pad
from ocnn.octree import Octree
from ocnn.utils import range_grid, trilinear_interp_weights


def adaptive_octree_sdf(octree: Octree, start_depth: int, threshold: float = 0.001):
  r'''Adaptively prune the octree based on the SDF interpolation error.
  '''

  depth = octree.depth
  assert start_depth >= octree.full_depth and start_depth < depth

  # the 27 interpolation weights for the 8 corners of a cube
  rng = range_grid(0, 2, device=octree.device) / 2.0
  weights = trilinear_interp_weights(rng[:, 0], rng[:, 1], rng[:, 2])
  corners = torch.tensor([0, 2, 6, 8, 18, 20, 24, 26], device=octree.device)

  # calcuate the interpolation error for each node at each depth.
  # the error is the max absolute difference between the original SDF value and
  # the interpolated SDF value from its parent node.
  keep = {}
  for d in range(start_depth, depth):
    fields_d = octree.fields[d + 1]    # !!! `d + 1` means children nodes
    if fields_d.dtype == torch.int16:  # quantized fields
      fields_d = fields_d.float() / octree.field_scale  # int16 -> float

    fields_c = fields_d[:, corners]    # (N, 8)
    interp_d = fields_c @ weights.t()  # (N, 27)
    error = (fields_d - interp_d).abs().max(dim=1)[0]  # (N)
    keep[d] = error > threshold

    # if no node in `depth-1` is kept, keep at least one, by combining with the
    # consistency check, this ensures that the octree is not empty after pruning
    if d == depth - 1 and keep[d].sum() == 0:
      i = torch.argmax(error)
      keep[d][i] = True

  # consistency check: if a node is kept, its parent node should also be kept
  for d in range(depth - 1, start_depth, -1):
    keep_c = octree_pad(keep[d].unsqueeze(1), octree, d, val=False)
    keep_c = keep_c.view(-1, 8).any(dim=1)
    keep[d - 1] = keep[d - 1] | keep_c

  # prune nodes based on the error threshold
  octree.prune(keep, start_depth)
  return octree
