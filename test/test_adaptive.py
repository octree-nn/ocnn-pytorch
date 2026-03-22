# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import unittest

import ocnn
from ocnn.octree.adaptive import adaptive_octree_sdf


class TestAdaptive(unittest.TestCase):

  def build_sphere_sdf_octree(self, depth=6, full_depth=4, radius=0.5):
    r''' Build an octree from a sphere SDF.
     '''

    resolution = 2 ** depth
    grid = torch.linspace(-1.0, 1.0, resolution)
    x, y, z = torch.meshgrid(grid, grid, grid, indexing='ij')
    sdf = torch.sqrt(x * x + y * y + z * z) - radius
    sdf = sdf.unsqueeze(0)  # (1, resolution, resolution, resolution)

    octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth)
    octree.build_octree_from_sdf(sdf, compress=False)
    return octree

  def build_plane_sdf_octree(self, depth=6, full_depth=4):
    r''' Build an octree from a planar SDF (z = 0).
     '''

    resolution = 2 ** depth
    grid = torch.linspace(-1.0, 1.0, resolution)
    _, _, z = torch.meshgrid(grid, grid, grid, indexing='ij')
    sdf = z  # Simple planar SDF
    sdf = sdf.unsqueeze(0)

    octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth)
    octree.build_octree_from_sdf(sdf, compress=False)
    return octree

  def test_adaptive_octree_sdf_basic(self):
    r''' Test basic functionality with a sphere SDF.
     '''

    octree = self.build_sphere_sdf_octree(depth=6, full_depth=3, radius=0.5)

    # Get initial node counts
    initial_nnum_nempty = octree.nnum_nempty.clone()

    # Apply adaptive pruning
    result = adaptive_octree_sdf(octree, start_depth=3, threshold=0.01)

    # With a high curvature sphere, some nodes should be pruned at deeper depths
    # (near the boundary where SDF changes rapidly)
    for d in range(octree.full_depth + 1, octree.depth + 1):
      # The result octree should have <= original number of non-empty nodes
      # (some nodes may be pruned)
      self.assertLessEqual(
          result.nnum_nempty[d].item(),
          initial_nnum_nempty[d].item(),
          f'Depth {d}: Nodes should not increase after pruning')


if __name__ == "__main__":
  unittest.main()
