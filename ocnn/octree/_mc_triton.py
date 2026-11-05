"""Optional Triton kernels; global edge deduplication stays in PyTorch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(F, S, COUNTS, V, CASE, FC, N: tl.constexpr, B: tl.constexpr):
  i = tl.program_id(0) * B + tl.arange(0, B)
  c = tl.arange(0, 8)
  slot = tl.load(S + i, i < N, 0)
  sibling = slot % 8
  base = 9 * (sibling >> 2) + 3 * ((sibling >> 1) & 1) + (sibling & 1)
  dx = (c == 1) | (c == 2) | (c == 5) | (c == 6)
  dy = (c == 2) | (c == 3) | (c == 6) | (c == 7)
  dz = c >= 4
  offset = dx.to(tl.int32) * 9 + dy.to(tl.int32) * 3 + dz.to(tl.int32)
  value = tl.load(F + (slot // 8)[:, None] * 27 + base[:, None] + offset[None, :],
                  i[:, None] < N, 1.)
  case = tl.sum(tl.where(value <= 0, 1 << c[None, :], 0), 1)
  fc = tl.load(COUNTS + case)
  tl.store(V + i[:, None] * 8 + c[None, :], value, i[:, None] < N)
  tl.store(CASE + i, case, i < N)
  tl.store(FC + i, fc, i < N)


@triton.jit
def _edges(X, CASE, PREFIX, A, EB, C, KEYS, OWNERS,
           N: tl.constexpr, SIZE: tl.constexpr, B: tl.constexpr):
  i = tl.program_id(0) * B + tl.arange(0, B)
  e = tl.arange(0, 16)
  a = tl.load(A + e, e < 12, 0)
  b = tl.load(EB + e, e < 12, 0)
  case = tl.load(CASE + i, i < N, 0)
  active = ((((case[:, None] >> a[None, :]) & 1) !=
             ((case[:, None] >> b[None, :]) & 1)) & (e[None, :] < 12))
  rank = tl.cumsum(active.to(tl.int32), 1) - 1
  prefix = tl.load(PREFIX + i, i < N, 0)
  x = tl.load(X + i * 3, i < N, 0)
  y = tl.load(X + i * 3 + 1, i < N, 0)
  z = tl.load(X + i * 3 + 2, i < N, 0)
  ax = tl.load(C + a * 3)
  ay = tl.load(C + a * 3 + 1)
  az = tl.load(C + a * 3 + 2)
  bx = tl.load(C + b * 3)
  by = tl.load(C + b * 3 + 1)
  axis = tl.where(bx != ax, 0, tl.where(by != ay, 1, 2))
  key = (((x[:, None] + ax[None, :]) * SIZE +
           y[:, None] + ay[None, :]) * SIZE + z[:, None] + az[None, :]) * 3
  key += axis[None, :]
  out = prefix[:, None] + rank
  mask = (i[:, None] < N) & active
  tl.store(KEYS + out, key, mask)
  tl.store(OWNERS + out, i[:, None] * 12 + e[None, :], mask)


@triton.jit
def _vertices(KEYS, OWNERS, V, A, EB, OUT,
              N: tl.constexpr, SIZE: tl.constexpr, B: tl.constexpr):
  i = tl.program_id(0) * B + tl.arange(0, B)
  key = tl.load(KEYS + i, i < N, 0)
  owner = tl.load(OWNERS + i, i < N, 0)
  edge, cell = owner % 12, owner // 12
  a, b = tl.load(A + edge), tl.load(EB + edge)
  va = tl.load(V + cell * 8 + a, i < N, 0.)
  vb = tl.load(V + cell * 8 + b, i < N, 1.)
  denom = vb - va
  t = tl.where(tl.abs(denom) < 1.e-30, 0.5, -va / denom)
  t = tl.minimum(tl.maximum(t, 0.), 1.)
  pos, axis = key // 3, key % 3
  x = (pos // (SIZE * SIZE)).to(tl.float32)
  y = ((pos // SIZE) % SIZE).to(tl.float32)
  z = (pos % SIZE).to(tl.float32)
  tl.store(OUT + i * 3, x + tl.where(axis == 0, t, 0.), i < N)
  tl.store(OUT + i * 3 + 1, y + tl.where(axis == 1, t, 0.), i < N)
  tl.store(OUT + i * 3 + 2, z + tl.where(axis == 2, t, 0.), i < N)


@triton.jit
def _faces(CASE, PREFIX, MAP, TABLE, OUT, N: tl.constexpr, B: tl.constexpr):
  i = tl.program_id(0) * B + tl.arange(0, B)
  j = tl.arange(0, 16)
  # Reverse each triangle's winding, matching cubvh's e0,e2,e1.
  local = j // 3 * 3 + tl.where(j % 3 == 0, 0, 3 - j % 3)
  case = tl.load(CASE + i, i < N, 0)
  edge = tl.load(TABLE + case[:, None] * 16 + local[None, :],
                 (i[:, None] < N) & (j[None, :] < 15), -1)
  mask = (i[:, None] < N) & (j[None, :] < 15) & (edge >= 0)
  vertex = tl.load(MAP + i[:, None] * 12 + edge, mask, 0)
  prefix = tl.load(PREFIX + i, i < N, 0)
  tl.store(OUT + prefix[:, None] * 3 + j[None, :], vertex, mask)


def gather(fields, slots, counts):
  n = slots.numel()
  values = torch.empty((n, 8), dtype=torch.float32, device=fields.device)
  cases = torch.empty(n, dtype=torch.long, device=fields.device)
  face_counts = torch.empty_like(cases)
  _gather[(triton.cdiv(n, 64),)](fields, slots, counts, values, cases,
                               face_counts, n, 64)
  return values, cases, face_counts


def edges(coords, cases, prefix, a, b, corners, size, count):
  keys = torch.empty(count, dtype=torch.long, device=coords.device)
  owners = torch.empty_like(keys)
  _edges[(triton.cdiv(cases.numel(), 32),)](
      coords, cases, prefix, a, b, corners, keys, owners,
      cases.numel(), size, 32)
  return keys, owners


def vertices(keys, owners, values, a, b, size):
  out = torch.empty((keys.numel(), 3), dtype=torch.float32, device=keys.device)
  _vertices[(triton.cdiv(keys.numel(), 128),)](
      keys, owners, values, a, b, out, keys.numel(), size, 128,
      enable_fp_fusion=False)
  return out


def faces(cases, prefix, edge_map, table, count):
  out = torch.empty((count, 3), dtype=torch.long, device=cases.device)
  _faces[(triton.cdiv(cases.numel(), 32),)](
      cases, prefix, edge_map, table, out, cases.numel(), 32)
  return out
