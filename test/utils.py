# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import numpy as np

import ocnn


def get_points(id, return_data=False):
  folder = os.path.dirname(__file__)
  filename = os.path.join(folder, 'data/octree/test_%03d.npz' % id)
  data = np.load(filename)

  points, normals = data['points'], data['normals']
  point_cloud = ocnn.octree.Points(
      torch.from_numpy(points), torch.from_numpy(normals))
  return (point_cloud, data) if return_data else point_cloud


def get_octree(id, return_data=False):
  point_cloud, data = get_points(id, return_data=True)
  octree = ocnn.octree.Octree(
      data['depth'].item(), full_depth=data['full_depth'].item())
  octree.build_octree(point_cloud)
  return (octree, data) if return_data else octree


def get_batch_octree(device='cpu'):
  octree1 = get_octree(4).to(device)
  octree2 = get_octree(5).to(device)
  octree = ocnn.octree.merge_octrees([octree1, octree2])
  octree.construct_all_neigh()
  return octree


def sphere_coords(resolution, device='cuda'):
  r''' This function generates random features and integer coordinates for
  voxels on a thin spherical shell inside a cubic grid of resolution
  `res`. It iterates in n^3 chunks to keep memory bounded, building 3D
  meshes via `torch.meshgrid` and shifting them into global coordinates.

  Args:
    resolution: int
      The resolution of the cubic grid.
    device: str
      The device where the tensors are allocated.
  '''

  n = 128
  out = []
  for i in range(0, resolution, n):
    for j in range(0, resolution, n):
      for k in range(0, resolution, n):
        block = torch.stack(torch.meshgrid(
            torch.arange(i, min(i + n, resolution), device=device),
            torch.arange(j, min(j + n, resolution), device=device),
            torch.arange(k, min(k + n, resolution), device=device),
            indexing='ij'), dim=-1).int()
        dist = ((block.float() - resolution / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
        active = (dist <= resolution / 2) & (dist >= resolution / 2 - 1.25)
        out.append(block[active])
  pos = torch.cat(out, dim=0)
  return pos
