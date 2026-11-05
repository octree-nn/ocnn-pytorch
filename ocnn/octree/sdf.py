from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
  from .octree import Octree


class SdfQueryResult(NamedTuple):
  sdf: torch.Tensor
  sign: torch.Tensor
  has_sdf: torch.Tensor
  valid: torch.Tensor


def _origin_column(index):
  sibling = index % 8
  return 9 * (sibling >> 2) + 3 * ((sibling >> 1) & 1) + (sibling & 1)


def _interpolate(fields, index, local):
  """Pair corners along z, y, x without a (Q, 8) tensor reduction."""
  row, base = index // 8, _origin_column(index)
  values = [fields[row, base + offset] for offset in (0, 1, 3, 4, 9, 10, 12, 13)]
  x, y, z = local.unbind(dim=1)
  for weight in (z, y, x):
    values = [a * (1 - weight) + b * weight
              for a, b in zip(values[::2], values[1::2])]
  return values[0]


def _query_corner_sign(octree: 'Octree', xyz, batch):
  """Query integer corners in [0, N]^3 of an unpruned tree, without interpolation."""
  depth, full_depth = octree.depth, octree.full_depth
  cell = xyz.clamp(max=2 ** depth - 1)
  coarse = cell >> (depth - full_depth)
  idx = batch * (8 ** full_depth)
  cx, cy, cz = coarse.unbind(dim=1)
  for bit in range(full_depth):
    idx = idx | (((cx >> bit) & 1) << (3 * bit + 2))
    idx = idx | (((cy >> bit) & 1) << (3 * bit + 1))
    idx = idx | (((cz >> bit) & 1) << (3 * bit))

  positive = torch.zeros_like(batch, dtype=torch.bool)
  active = torch.ones_like(positive)
  for d in range(full_depth, depth):
    if octree.children[d].numel() == 0:
      break
    idx = torch.where(active, idx, 0)
    child = octree.children[d][idx].long()
    if d == full_depth:
      value = octree.fields[d][idx]
    else:
      value = octree.fields[d][idx // 8, _origin_column(idx)]
    positive = torch.where(active & (child < 0), value > 0, positive)
    active = active & (child >= 0)
    bits = (cell >> (depth - d - 1)) & 1
    idx = 8 * child + 4 * bits[:, 0] + 2 * bits[:, 1] + bits[:, 2]

  if octree.fields[depth].numel() != 0:
    idx = torch.where(active, idx, 0)
    dx, dy, dz = (xyz - cell).unbind(dim=1)
    column = _origin_column(idx) + 9 * dx + 3 * dy + dz
    value = octree.fields[depth][idx // 8, column]
    positive = torch.where(active, value > 0, positive)
  return torch.where(positive, 1.0, -1.0)


def octree_query_sdf(octree: 'Octree', pts: torch.Tensor) -> SdfQueryResult:
  r'''Queries an unpruned, unquantized SDF octree with batched points.

  Args:
    octree (Octree): Built by :meth:`Octree.build_octree_from_sdf` with
        ``compress=False`` and ``full_depth < depth``. Adaptive pruning is not
        supported: its terminal cells need not have a constant sign.
    pts (torch.Tensor): Floating-point tensor of shape ``(Q, 4)``, containing
        ``(x, y, z, batch)`` on the octree device. Coordinates are in the input
        SDF's sample-index space, **not** world space. A depth-D tree has
        ``N=2**D`` cells and input shape ``(B, N+1, N+1, N+1)``; its valid
        domain is the half-open box ``[0, N)^3``. Each point belongs only to
        the cell selected by ``floor``; neighboring cells are not queried.

  Returns:
    SdfQueryResult: Four tensors of shape ``(Q,)``. ``sdf`` contains trilinear
    interpolation of the original samples where a finest-level node slot
    exists (including empty siblings), and NaN elsewhere. ``has_sdf`` marks
    those values. ``sign`` is +1/-1 for sign-only results, and the sign of the
    interpolated value (including 0) otherwise. Invalid points have
    ``valid=False``, ``has_sdf=False``, ``sign=0`` and ``sdf=NaN``.

  Uses only PyTorch operations, with batched gathers and a depth loop. Reuse
  ``torch.compile(octree_query_sdf, fullgraph=True)`` for repeated CPU/CUDA
  queries; warm up before timing to exclude compilation. No custom kernel,
  spatial search or dense reconstruction is needed.
  '''

  depth, full_depth = octree.depth, octree.full_depth
  if not 0 <= full_depth < depth:
    raise ValueError('SDF queries require 0 <= full_depth < depth.')
  if pts.ndim != 2 or pts.shape[1] != 4 or not pts.is_floating_point():
    raise ValueError('pts must be a floating-point tensor of shape (Q, 4).')
  field = octree.fields[full_depth]
  if field is None or not field.is_floating_point():
    raise ValueError('Build an SDF octree with compress=False before querying.')
  if pts.device != field.device:
    raise ValueError('pts and the octree must be on the same device.')

  size = 2 ** depth
  xyz, batch = pts[:, :3], pts[:, 3]
  valid = ((xyz >= 0).all(dim=1) & (xyz < size).all(dim=1) &
           (batch >= 0) & (batch < octree.batch_size) &
           (batch == batch.floor()))
  # Invalid lanes still participate in gathers, using safe placeholder indices.
  safe = torch.where(valid.unsqueeze(1), pts, torch.zeros_like(pts))
  cell = safe[:, :3].floor().long()
  coarse = cell >> (depth - full_depth)
  idx = safe[:, 3].long() * (8 ** full_depth)
  # Small coarse grids need only a few bits; direct interleaving avoids three
  # dependent lookup-table gathers in the compiled query kernel.
  cx, cy, cz = coarse.unbind(dim=1)
  for bit in range(full_depth):
    idx = idx | (((cx >> bit) & 1) << (3 * bit + 2))
    idx = idx | (((cy >> bit) & 1) << (3 * bit + 1))
    idx = idx | (((cz >> bit) & 1) << (3 * bit))

  sdf = pts.new_full((pts.shape[0],), float('nan'),
                     dtype=torch.promote_types(pts.dtype, field.dtype))
  sign = torch.zeros(pts.shape[0], device=pts.device, dtype=torch.int8)
  active = valid

  for d in range(full_depth, depth):
    # Empty layers occur for constant-sign input fields. This is a tensor-shape
    # check, not a GPU scalar read or a data-dependent Python condition.
    if octree.children[d].numel() == 0:
      break
    idx = torch.where(active, idx, torch.zeros_like(idx))
    child = octree.children[d][idx].long()
    stop = active & (child < 0)
    if d == full_depth:
      value = field[idx]
    else:
      value = octree.fields[d][idx // 8, _origin_column(idx)]
    node_sign = torch.where(value > 0, 1, -1).to(torch.int8)
    sign = torch.where(stop, node_sign, sign)
    active = active & (child >= 0)

    bits = (cell >> (depth - d - 1)) & 1
    sibling = 4 * bits[:, 0] + 2 * bits[:, 1] + bits[:, 2]
    idx = 8 * child + sibling

  has_sdf = active
  if octree.fields[depth].numel() != 0:
    idx = torch.where(active, idx, torch.zeros_like(idx))
    value = _interpolate(octree.fields[depth], idx,
                         safe[:, :3] - cell.to(pts.dtype))
    sdf = torch.where(active, value, sdf)
    sign = torch.where(active, value.sign().to(torch.int8), sign)

  return SdfQueryResult(sdf, sign, has_sdf, valid)
