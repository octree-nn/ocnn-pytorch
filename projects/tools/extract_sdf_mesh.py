"""Export meshes from a saved, non-adaptive SDF octree.

Run from the repository root with PYTHONPATH=. python
projects/tools/extract_sdf_mesh.py input.pt output.obj --device cuda:0
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ocnn.octree import octree_marching_cubes
from ocnn.octree.sdf_io import load_sdf_octree


def write_obj(path, mesh):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w') as output:
    np.savetxt(output, mesh.vertices.cpu().numpy(), fmt='v %.9g %.9g %.9g')
    np.savetxt(output, mesh.faces.cpu().numpy() + 1, fmt='f %d %d %d')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('input', type=Path)
  parser.add_argument('output', type=Path)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--backend', choices=('auto', 'torch', 'triton'),
                      default='auto')
  parser.add_argument('--sample-coordinates', action='store_true',
                      help='Keep sample indices instead of mapping to [-1,1]')
  args = parser.parse_args()
  if args.output.suffix.lower() != '.obj':
    parser.error('Output must be an .obj file')
  tree, _ = load_sdf_octree(args.input, device=args.device)
  device = torch.device(args.device)
  if device.type == 'cuda':
    torch.cuda.synchronize(device)
  start = perf_counter()
  meshes = octree_marching_cubes(tree, backend=args.backend,
                                 world=not args.sample_coordinates)
  if device.type == 'cuda':
    torch.cuda.synchronize(device)
  elapsed = perf_counter() - start
  paths = []
  for batch, mesh in enumerate(meshes):
    path = (args.output if len(meshes) == 1 else args.output.with_name(
        f'{args.output.stem}-b{batch:03d}.obj'))
    write_obj(path, mesh)
    paths.append({'path': str(path), 'vertices': mesh.vertices.shape[0],
                  'faces': mesh.faces.shape[0]})
  print(json.dumps({'extraction_seconds': elapsed, 'outputs': paths}, indent=2))


if __name__ == '__main__':
  main()
