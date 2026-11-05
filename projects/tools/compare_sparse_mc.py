"""Compare octree sparse MC against pinned cubvh on real ShapeNet data.

Inputs are read only. Output contains raw extracted meshes, topology/geometry
checks and synchronized wall-clock timings excluding disk I/O and compilation.
The version-2 loader here is limited to extraction; it does not reinterpret
legacy query-domain metadata for SDF queries.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import cubvh
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import torch
import trimesh

from ocnn.octree import Octree, octree_marching_cubes


DEFAULT_SAMPLES = (
    '02691156/10155655850468db78d106ce0a280f87',
    '03001627/1006be65e7bc937e9141f9b58470d646',
    '02958343/1005ca47e516495512da0dbf3c68e847',
    '02958343/88ad6f5df57cf8ea16abce8cb03e7794',
)


def load_extraction_tree(path, device):
  data = torch.load(path, map_location='cpu', weights_only=True)
  depth, full = data['depth'], data['full_depth']
  if (data['format'] != 'ocnn.sdf_octree' or data['version'] not in (2, 3) or
      data['adaptive'] or data['quantized'] or data['batch_size'] != 1 or
      data['coordinate_convention'] != 'corner_sample_index' or
      data['sample_shape'] != [1] + [2**depth + 1] * 3):
    raise ValueError('Expected a version-2/3 nonadaptive FP32 corner SDF tree')
  tree = Octree(depth, full)
  for name in ('keys', 'children', 'fields', 'nnum', 'nnum_nempty'):
    setattr(tree, name, data[name])
  for d in range(full + 1, depth + 1):
    if (tree.fields[d].dtype != torch.float32 or
        tree.fields[d].shape != (int(tree.nnum_nempty[d-1]), 27)):
      raise ValueError(f'Invalid field layout at depth {d}')
  return tree.to(device), data


def prepare(tree):
  slot = tree.nempty_index(tree.depth)
  x, y, z, _ = tree.xyzb(tree.depth, nempty=True)
  coords = torch.stack((x, y, z), 1).int().contiguous()
  sibling = slot % 8
  base = 9 * (sibling >> 2) + 3 * ((sibling >> 1) & 1) + (sibling & 1)
  offsets = slot.new_tensor([0, 9, 12, 3, 1, 10, 13, 4])
  values = tree.fields[tree.depth][slot[:, None] // 8,
                                   base[:, None] + offsets]
  return coords, values.contiguous()


def timed(function, repeats):
  for _ in range(3):
    function()
  torch.cuda.synchronize()
  samples = []
  for _ in range(repeats):
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    samples.append((time.perf_counter() - start) * 1000)
    del result
  return {'median_ms': statistics.median(samples),
          'min_ms': min(samples), 'max_ms': max(samples), 'samples_ms': samples}


def numpy_mesh(result, scale):
  vertices, faces = result
  return trimesh.Trimesh(vertices.cpu().numpy() * scale - 1,
                         faces.cpu().numpy(), process=False)


def topology(mesh):
  f = np.asarray(mesh.faces)
  area = mesh.area_faces
  edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
  _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
  return {'vertices': len(mesh.vertices), 'faces': len(f),
          'finite': bool(np.isfinite(mesh.vertices).all()),
          'zero_area_faces': int((area == 0).sum()),
          'boundary_edges': int((counts == 1).sum()),
          'nonmanifold_edges': int((counts > 2).sum()),
          'watertight': bool(mesh.is_watertight),
          'winding_consistent': bool(mesh.is_winding_consistent),
          'bounds': mesh.bounds.tolist()}


def compare_meshes(first, second):
  distance, mapping = cKDTree(first.vertices).query(second.vertices, workers=4)
  reverse, _ = cKDTree(second.vertices).query(first.vertices, workers=4)
  faces = np.sort(first.faces, axis=1)
  mapped = np.sort(mapping[second.faces], axis=1)
  faces = faces[np.lexsort(faces.T[::-1])]
  mapped = mapped[np.lexsort(mapped.T[::-1])]
  return {'max_vertex_distance_world': float(max(distance.max(), reverse.max())),
          'one_to_one_vertex_mapping': bool(len(np.unique(mapping)) ==
                                            len(first.vertices) == len(second.vertices)),
          'same_unoriented_triangles': bool(faces.shape == mapped.shape and
                                             np.array_equal(faces, mapped))}


def distance_to_source(source, extracted, samples=10000):
  def direction(query_mesh, target_mesh):
    points, _ = trimesh.sample.sample_surface(query_mesh, samples, seed=42)
    bvh = cubvh.cuBVH(np.asarray(target_mesh.vertices, dtype=np.float32),
                      np.asarray(target_mesh.faces, dtype=np.uint32))
    distances, _, _ = bvh.unsigned_distance(torch.as_tensor(
        points, dtype=torch.float32, device='cuda'))
    values = distances.cpu().numpy()
    finite = values[np.isfinite(values)]
    return {'samples': samples, 'nonfinite': int((~np.isfinite(values)).sum()),
            'mean': float(np.mean(finite)), 'p95': float(np.quantile(finite, .95)),
            'max_sampled': float(np.max(finite))}
  return {'extracted_to_source': direction(extracted, source),
          'source_to_extracted': direction(source, extracted)}


def surface_residual(mesh, dense, size):
  positions = (np.asarray(mesh.vertices) + 1) * (size/2)
  values = map_coordinates(dense, positions.T, order=1,
                            mode='constant', cval=np.nan)
  return {'max_abs_sdf': float(np.nanmax(np.abs(values))),
          'nonfinite': int((~np.isfinite(values)).sum())}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset-root', type=Path, required=True)
  parser.add_argument('--source-root', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--samples', nargs='+', default=DEFAULT_SAMPLES)
  parser.add_argument('--repeats', type=int, default=20)
  args = parser.parse_args()
  torch.set_num_threads(4)
  args.output.mkdir(parents=True, exist_ok=True)
  report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'cubvh_commit': 'de1badf89b54975d407c42a02a47d41fa149e486',
            'timing': '3 warmups then synchronized wall time; disk and JIT excluded',
            'samples': []}
  for sample in args.samples:
    category, model = sample.split('/')
    out = args.output / category / model
    out.mkdir(parents=True, exist_ok=True)
    tree, payload = load_extraction_tree(
        args.dataset_root/'octree-sdf'/f'{sample}.pt', 'cuda')
    coords, values = prepare(tree)
    size = 2**tree.depth
    dense_path = args.dataset_root/'sdf'/f'{sample}.npy'
    if not dense_path.exists():
      dense_path = args.source_root/'r256/sdf'/f'{sample}.npy'
    dense = np.load(dense_path)
    padded = np.pad(dense, ((0, 1),) * 3, constant_values=1.)
    offsets = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                        [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    corners = coords.cpu().numpy()[:, None, :] + offsets[None, :, :]
    expected = padded[corners[..., 0], corners[..., 1], corners[..., 2]]
    corner_error = float(np.max(np.abs(values.cpu().numpy() - expected)))
    if corner_error != 0:
      raise AssertionError(f'{sample}: octree and dense corner mismatch {corner_error}')
    source = trimesh.load(args.source_root/'ShapeNetCore.v1'/sample/'model.obj',
                          force='mesh', process=False, skip_materials=True)
    bbox = np.load(args.source_root/'r256/bbox'/f'{sample}.npz')
    low, high = bbox['bbmin'], bbox['bbmax']
    if not np.allclose(source.bounds, [low, high], atol=1.e-5):
      raise AssertionError('Original mesh bounds disagree with normalization metadata')
    source.vertices = ((source.vertices - (low + high)/2) *
                       (2 * float(bbox['mul']) / (high-low).max()))
    source.export(out/'source.ply')

    def ours():
      return octree_marching_cubes(tree, backend='triton')[0]

    def theirs_core():
      return cubvh.sparse_marching_cubes(coords, values, 0., False)

    def theirs_total():
      c, v = prepare(tree)
      return cubvh.sparse_marching_cubes(c, v, 0., False)

    timings = {name: timed(fn, args.repeats) for name, fn in (
        ('ours_octree_total', ours), ('cubvh_core', theirs_core),
        ('cubvh_octree_total', theirs_total))}
    first = numpy_mesh(ours(), 2/size)
    second = numpy_mesh(theirs_core(), 2/size)
    first.export(out/'triton.ply')
    second.export(out/'cubvh.ply')
    zeros = int((values == 0).sum().item())
    entry = {'sample': sample, 'format_version': payload['version'],
             'cells': len(coords), 'dense_path': str(dense_path),
             'corner_max_error': corner_error, 'zero_corner_instances': zeros,
             'timings': timings, 'ours': topology(first), 'cubvh': topology(second),
             'mesh_comparison': compare_meshes(first, second),
             'dense_surface_residual': surface_residual(first, padded, size),
             'source_distance': distance_to_source(source, first)}
    if zeros:
      # Align cubvh's strict <0 with the octree's <=0 convention for a
      # diagnostic only; never change the stored tree or native benchmark.
      aligned_values = torch.where(values == 0, -torch.finfo(torch.float32).tiny,
                                    values)
      aligned = numpy_mesh(cubvh.sparse_marching_cubes(
          coords, aligned_values, 0., False), 2/size)
      aligned.export(out/'cubvh-zero-aligned.ply')
      entry['cubvh_zero_aligned'] = topology(aligned)
      entry['aligned_mesh_comparison'] = compare_meshes(first, aligned)
    (out/'checks.json').write_text(json.dumps(entry, indent=2)+'\n')
    report['samples'].append(entry)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(entry), flush=True)
    del tree, coords, values, dense, padded


if __name__ == '__main__':
  main()
