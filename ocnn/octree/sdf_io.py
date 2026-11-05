"""Read and write SDF octrees."""

from pathlib import Path

import torch

from .octree import Octree

__all__ = ['load_sdf_octree', 'save_sdf_octree']

_PROPERTIES = ('keys', 'children', 'fields', 'nnum', 'nnum_nempty')


def load_sdf_octree(filename, device='cpu'):
  """Return ``(tree, metadata)``. Query coordinates use ``[0, 2**depth)^3``."""
  payload = torch.load(filename, map_location='cpu', weights_only=True)
  tree = Octree(payload['depth'], payload['full_depth'], payload['batch_size'])
  tree.field_scale = payload['field_scale']
  for name in _PROPERTIES:
    setattr(tree, name, payload[name])
  for d in range(tree.depth + 1):
    batch_id = tree.keys[d] >> 48
    tree.batch_nnum[d] = torch.bincount(batch_id, minlength=tree.batch_size)
    tree.batch_nnum_nempty[d] = torch.bincount(
        batch_id[tree.children[d] >= 0], minlength=tree.batch_size)
  return tree.to(device), payload['metadata']


def save_sdf_octree(octree, filename, metadata=None):
  """Save the tree and return the file size in bytes."""
  tree = octree.cpu()
  payload = {
      'depth': tree.depth,
      'full_depth': tree.full_depth, 'batch_size': tree.batch_size,
      'field_scale': tree.field_scale,
      'metadata': {} if metadata is None else metadata,
      **{name: getattr(tree, name) for name in _PROPERTIES},
  }
  torch.save(payload, filename)
  return Path(filename).stat().st_size
