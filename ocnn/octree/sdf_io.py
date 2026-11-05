"""Read and write unpruned float32 SDF octrees."""

import os
import tempfile
from pathlib import Path

import torch

from .octree import Octree

__all__ = ['load_sdf_octree', 'save_sdf_octree']

FORMAT = 'ocnn.sdf_octree'
VERSION = 3
QUERY_DOMAIN = 'half_open'
_PROPERTIES = ('keys', 'children', 'fields', 'nnum', 'nnum_nempty')


def _validate(payload):
  if payload.get('format') != FORMAT or payload.get('version') != VERSION:
    raise ValueError('Unsupported SDF octree format or version')
  depth = payload['depth']
  full_depth = payload['full_depth']
  batch_size = payload['batch_size']
  if not (0 <= full_depth < depth and depth > 4 and batch_size > 0):
    raise ValueError('Invalid depth, full_depth or batch_size')
  if payload['adaptive'] or payload['quantized']:
    raise ValueError('Expected unpruned float32 SDF fields')
  if (payload.get('sample_shape') != [batch_size] + [2 ** depth + 1] * 3 or
      payload.get('query_domain') != QUERY_DOMAIN or
      payload.get('query_upper_bound') != 2 ** depth or
      payload.get('coordinate_convention') != 'corner_sample_index'):
    raise ValueError('Invalid sample-coordinate metadata')
  for d in range(depth + 1):
    node_count = int(payload['nnum'][d])
    keys = payload['keys'][d]
    children = payload['children'][d]
    field = payload['fields'][d]
    if (keys.shape != (node_count,) or keys.dtype != torch.int64 or
        children.shape != (node_count,) or children.dtype != torch.int32):
      raise ValueError(f'Invalid topology at depth {d}')
    if d < full_depth:
      if field is not None:
        raise ValueError(f'Unexpected field at depth {d}')
      continue
    if d == full_depth:
      shape = (node_count,)
    else:
      parent_count = int(payload['nnum_nempty'][d - 1])
      shape = (parent_count, 27)
    if field is None or field.shape != shape or field.dtype != torch.float32:
      raise ValueError(f'Invalid float32 SDF field at depth {d}')


def load_sdf_octree(filename, device='cpu'):
  """Return ``(tree, metadata)``. Query coordinates use ``[0, 2**depth)^3``."""
  payload = torch.load(filename, map_location='cpu', weights_only=True)
  _validate(payload)
  tree = Octree(payload['depth'], payload['full_depth'], payload['batch_size'])
  tree.field_scale = payload['field_scale']
  for name in _PROPERTIES:
    setattr(tree, name, payload[name])
  for d in range(tree.depth + 1):
    batch_id = tree.keys[d] >> 48
    tree.batch_nnum[d] = torch.bincount(batch_id, minlength=tree.batch_size)
    tree.batch_nnum_nempty[d] = torch.bincount(
        batch_id[tree.children[d] >= 0], minlength=tree.batch_size)
  return tree.to(device), payload.get('metadata', {})


def save_sdf_octree(octree, filename, metadata=None):
  """Atomically save the tree and return the file size in bytes."""
  tree = octree.cpu()
  payload = {
      'format': FORMAT, 'version': VERSION, 'depth': tree.depth,
      'full_depth': tree.full_depth, 'batch_size': tree.batch_size,
      'field_scale': tree.field_scale,
      'sample_shape': [tree.batch_size] + [2 ** tree.depth + 1] * 3,
      'coordinate_convention': 'corner_sample_index',
      'query_upper_bound': 2 ** tree.depth, 'query_domain': QUERY_DOMAIN,
      'adaptive': False, 'quantized': False, 'metadata': metadata or {},
      **{name: getattr(tree, name) for name in _PROPERTIES},
  }
  _validate(payload)
  filename = Path(filename)
  filename.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(dir=filename.parent, suffix='.tmp')
  os.close(descriptor)
  try:
    torch.save(payload, temporary)
    os.replace(temporary, filename)
  finally:
    Path(temporary).unlink(missing_ok=True)
  return filename.stat().st_size
