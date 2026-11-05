"""Compare octree SDF queries with dense grid_sample interpolation.

Run from the repository root, for example:
  PYTHONPATH=. python projects/tools/benchmark_sdf_query.py --depth 7 \
      --device cuda:0 --compile-torch
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer

from ocnn.octree import Octree, octree_query_sdf


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--depth', type=int, default=6)
  parser.add_argument('--full-depth', type=int, default=3)
  parser.add_argument('--queries', type=int, default=100000)
  parser.add_argument('--min-run-time', type=float, default=0.3)
  parser.add_argument('--compile-torch', action='store_true')
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--threads', type=int, default=4)
  parser.add_argument('--output', type=Path)
  args = parser.parse_args()
  if not (args.depth > 4 and 0 <= args.full_depth < args.depth and
          args.queries > 0 and args.min_run_time > 0 and args.threads > 0):
    parser.error('Require depth > full-depth >= 0, depth > 4, and positive '
                 'queries, min-run-time and threads.')
  torch.set_num_threads(args.threads)
  torch.manual_seed(42)
  device = torch.device(args.device)
  size = 2 ** args.depth
  axis = torch.linspace(-1, 1, size + 1, device=device)
  xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), dim=-1)
  # Enclosed shell: exterior and cavity are positive, material is negative.
  volume = ((xyz.norm(dim=-1) - 0.26).abs() - 0.06).unsqueeze(0)
  octree = Octree(args.depth, args.full_depth, device=device)
  octree.build_octree_from_sdf(volume)

  def dense_query(pts):
    grid = pts[:, [2, 1, 0]] * (2.0 / size) - 1
    return F.grid_sample(volume[:, None], grid[None, :, None, None, :],
                         align_corners=True).reshape(-1)

  query = (torch.compile(octree_query_sdf, fullgraph=True)
            if args.compile_torch else octree_query_sdf)

  def timed(func):
    # Timer warms up and synchronizes accelerators. Report the median of
    # repeated measurements, excluding one-time compilation and tree building.
    measurement = Timer('func()', globals={'func': func},
                        num_threads=args.threads).blocked_autorange(
                            min_run_time=args.min_run_time)
    return measurement.median * 1000

  report = {
      'torch': torch.__version__, 'device': str(device),
      'threads': args.threads,
      'depth': args.depth, 'full_depth': args.full_depth,
      'volume_shape': list(volume.shape),
      'queries_per_distribution': args.queries,
      'min_run_time': args.min_run_time, 'seed': 42,
      'compiled': args.compile_torch,
      'dense_bytes': volume.numel() * volume.element_size(),
      'fields_bytes': sum(f.numel() * f.element_size()
                          for f in octree.fields if f is not None),
      'keys_children_bytes': sum(t.numel() * t.element_size()
                                 for seq in (octree.keys, octree.children)
                                 for t in seq if t is not None),
      'distributions': {},
  }
  for distribution in ('uniform', 'near_surface'):
    pts = torch.zeros(args.queries, 4, device=device)
    if distribution == 'uniform':
      pts[:, :3] = torch.rand(args.queries, 3, device=device) * size
    else:
      direction = F.normalize(
          torch.randn(args.queries, 3, device=device), dim=1)
      radius = torch.where(torch.rand(args.queries, 1, device=device) > 0.5,
                           0.32, 0.20)
      radius = radius + (torch.rand_like(radius) - 0.5) * (4.0 / size)
      pts[:, :3] = (direction * radius + 1) * (size / 2)
    result = query(octree, pts)
    reference = dense_query(pts)
    error = (result.sdf[result.has_sdf] - reference[result.has_sdf]).abs()
    sign_only = ~result.has_sdf
    sign_match = (result.sign > 0) == (reference > 0)
    if not result.valid.all() or not sign_match.all():
      raise AssertionError('Validity or sign comparison failed.')
    torch.testing.assert_close(result.sdf[result.has_sdf],
                               reference[result.has_sdf], atol=2.e-6, rtol=0)
    row = {
        'has_sdf': int(result.has_sdf.sum()),
        'sign_only': int(sign_only.sum()),
        'max_abs_sdf_error': float(error.max()) if error.numel() else None,
        'mean_abs_sdf_error': float(error.mean()) if error.numel() else None,
        'all_sign_mismatches': int((~sign_match).sum()),
        'sign_only_mismatches': int((~sign_match & sign_only).sum()),
    }
    del result, reference, error, sign_only, sign_match
    row['octree_ms'] = timed(lambda pts=pts: query(octree, pts))
    row['eager_torch_ms'] = (
        timed(lambda pts=pts: octree_query_sdf(octree, pts))
        if args.compile_torch else row['octree_ms'])
    row['dense_ms'] = timed(lambda pts=pts: dense_query(pts))
    report['distributions'][distribution] = row
  text = json.dumps(report, indent=2)
  print(text)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + '\n')


if __name__ == '__main__':
  with torch.no_grad():
    main()
