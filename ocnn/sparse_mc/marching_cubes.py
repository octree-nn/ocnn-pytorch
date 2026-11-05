"""Classic sparse Marching Cubes on unpruned, corner-sampled SDF octrees."""

from functools import lru_cache
from typing import NamedTuple

import torch

from ._mc_table import TRI_TABLE


class OctreeMesh(NamedTuple):
  vertices: torch.Tensor
  faces: torch.Tensor


# Edges always run in the positive axis direction, including MC edges 2/3/6/7.
_CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
_EDGE_A = (0, 1, 3, 0, 4, 5, 7, 4, 0, 1, 2, 3)
_EDGE_B = (1, 2, 2, 3, 5, 6, 6, 7, 4, 5, 6, 7)


@lru_cache(maxsize=16)
def _tables(device):
  corners = torch.tensor(_CORNERS, device=device)
  a = torch.tensor(_EDGE_A, device=device)
  b = torch.tensor(_EDGE_B, device=device)
  triangles = torch.tensor(TRI_TABLE, dtype=torch.long, device=device)
  counts = (triangles >= 0).sum(1) // 3
  return corners, a, b, triangles, counts


def _empty(device):
  return OctreeMesh(torch.empty((0, 3), dtype=torch.float32, device=device),
                    torch.empty((0, 3), dtype=torch.long, device=device))


def _gather_torch(fields, slots, table, counts):
  sibling = slots % 8
  base = 9 * (sibling >> 2) + 3 * ((sibling >> 1) & 1) + (sibling & 1)
  offsets = table[:, 0] * 9 + table[:, 1] * 3 + table[:, 2]
  values = fields[slots[:, None] // 8, base[:, None] + offsets]
  # Match build_octree_from_sdf's >0 partition: exact zero is inside.
  cases = ((values <= 0).long() << torch.arange(8, device=fields.device)).sum(1)
  return values, cases, counts[cases]


def _extract(fields, slots, coords, size, backend):
  device = fields.device
  corners, a, b, triangles, counts = _tables(str(device))
  kernels = None
  if backend == 'triton':
    from . import _mc_triton as kernels
    values, cases, face_counts = kernels.gather(fields, slots, counts)
  else:
    values, cases, face_counts = _gather_torch(fields, slots, corners, counts)
  edge_mask = ((cases[:, None] >> a) & 1) != ((cases[:, None] >> b) & 1)
  edge_counts = edge_mask.sum(1)
  edge_prefix = edge_counts.cumsum(0) - edge_counts
  face_prefix = face_counts.cumsum(0) - face_counts
  num_edges = int(edge_counts.sum().item())
  num_faces = int(face_counts.sum().item())
  if num_edges == 0:
    return _empty(device)

  if kernels is not None:
    keys, owners = kernels.edges(coords, cases, edge_prefix, a, b, corners,
                                 size, num_edges)
  else:
    owners = edge_mask.flatten().nonzero().flatten()
    cell, edge = owners // 12, owners % 12
    start = coords[cell] + corners[a[edge]]
    axis = (corners[b[edge]] - corners[a[edge]]).argmax(1)
    keys = ((start[:, 0] * size + start[:, 1]) * size + start[:, 2]) * 3 + axis

  unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
  # Select one cell per edge. Consistent fields and canonical interpolation
  # direction make the position independent of the representative cell.
  representatives = torch.full_like(unique_keys, slots.numel() * 12)
  representatives.scatter_reduce_(0, inverse, owners, reduce='amin')
  edge_map = torch.full((slots.numel() * 12,), -1, dtype=torch.long,
                        device=device)
  edge_map[owners] = inverse

  if kernels is not None:
    vertices = kernels.vertices(unique_keys, representatives, values, a, b, size)
    faces = kernels.faces(cases, face_prefix, edge_map, triangles, num_faces)
  else:
    cell, edge = representatives // 12, representatives % 12
    va, vb = values[cell, a[edge]], values[cell, b[edge]]
    denom = vb - va
    t = torch.where(denom.abs() < 1.e-30, 0.5, -va / denom).clamp(0, 1)
    position = unique_keys // 3
    vertices = torch.stack((position // (size * size),
                            (position // size) % size, position % size), 1).float()
    vertices[torch.arange(vertices.shape[0], device=device), unique_keys % 3] += t
    table = triangles[cases, :15].view(-1, 5, 3)
    valid = table[:, :, 0] >= 0
    cell = torch.arange(slots.numel(), device=device)[:, None, None]
    faces = edge_map[(cell * 12 + table.clamp_min(0))][valid][:, [0, 2, 1]]
  return OctreeMesh(vertices, faces)


@torch.no_grad()
def octree_marching_cubes(octree, *, backend='auto', world=False):
  r'''Extracts the zero isosurface without reconstructing a dense volume.

  Args:
    octree: Non-adaptive SDF octree built with ``compress=False``. Shared
        corner samples must agree. Only the finest nonempty cells are used.
    backend (str): ``'torch'`` for the CPU/CUDA reference, ``'triton'`` for
        CUDA kernels plus PyTorch global deduplication, or ``'auto'`` to use
        Triton on CUDA when installed and Torch otherwise.
    world (bool): Map sample-index coordinates to ``[-1, 1]`` using
        ``vertices * (2 / 2**depth) - 1``. Default: False.

  Returns:
    List[OctreeMesh]: One mesh per batch, including empty meshes. Vertices
    are float32 and faces int64 on the field device. Face winding points
    towards positive SDF for regular cases. Exact zero is classified inside,
    consistently with the tree's ``sdf > 0`` cell selection.

  This is classic fixed-table MC: no ambiguity resolution, degeneracy repair,
  adaptive transitions or autograd. Degenerate triangles are retained.
  Dynamic allocation/deduplication can synchronize CUDA; this API is not
  intended for fullgraph compilation or CUDA graph capture.
  '''
  if backend not in ('auto', 'torch', 'triton'):
    raise ValueError('backend must be auto, torch or triton')
  if not 0 <= octree.full_depth < octree.depth <= 16:
    raise ValueError('Requires full_depth < depth <= 16')
  if getattr(octree, '_sdf_pruned', False):
    raise ValueError('Adaptive/pruned SDF octrees are not supported')
  fields = octree.fields[octree.depth]
  if fields is None or fields.dtype != torch.float32:
    raise ValueError('Requires an unquantized float32 SDF octree')
  if fields.ndim != 2 or fields.shape[1] != 27:
    raise ValueError('Expected finest SDF fields with shape (parents, 27)')
  device = fields.device
  if backend == 'auto':
    from importlib.util import find_spec
    backend = ('triton' if device.type == 'cuda' and find_spec('triton')
               else 'torch')
  if backend == 'triton' and device.type != 'cuda':
    raise ValueError('The Triton backend requires CUDA fields')
  slots = octree.nempty_index(octree.depth)
  if slots.numel() == 0:
    return [_empty(device) for _ in range(octree.batch_size)]
  x, y, z, batch = octree.xyzb(octree.depth, nempty=True)
  coords = torch.stack((x, y, z), 1).long()
  if fields.shape[0] * 8 != octree.keys[octree.depth].numel():
    raise ValueError('Finest SDF fields do not match the octree topology')
  meshes = []
  for bs in range(octree.batch_size):
    selected = batch == bs
    mesh = (_extract(fields, slots[selected], coords[selected],
                     2 ** octree.depth + 1, backend)
            if selected.any() else _empty(device))
    if world:
      mesh = OctreeMesh(mesh.vertices * (2.0 / 2 ** octree.depth) - 1,
                        mesh.faces)
    meshes.append(mesh)
  return meshes
