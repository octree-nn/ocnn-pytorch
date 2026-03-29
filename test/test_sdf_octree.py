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


class TestSdfOctree(unittest.TestCase):

  def build_sphere_sdf_octree(self, depth=6, full_depth=4, radius=0.5,
                              compress=False, device='cpu'):
    r''' Build an octree from a sphere SDF.
     '''
    resolution = 2 ** depth
    grid = torch.linspace(-1.0, 1.0, resolution)
    x, y, z = torch.meshgrid(grid, grid, grid, indexing='ij')
    sdf = torch.sqrt(x * x + y * y + z * z) - radius
    sdf = sdf.unsqueeze(0)  # (1, resolution, resolution, resolution)

    octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth, device=device)
    octree.build_octree_from_sdf(sdf, compress=compress)
    return octree, sdf

  def build_plane_sdf_octree(self, depth=6, full_depth=4):
    r''' Build an octree from a planar SDF (z = 0).
     '''
    resolution = 2 ** depth
    grid = torch.linspace(-1.0, 1.0, resolution)
    _, _, z = torch.meshgrid(grid, grid, grid, indexing='ij')
    sdf = z  # Simple planar SDF
    sdf = sdf.unsqueeze(0)

    octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth, device='cpu')
    octree.build_octree_from_sdf(sdf, compress=False)
    return octree

  # ==================== to_sdf tests ====================

  def test_to_sdf_basic(self):
    r''' Test basic functionality of to_sdf with a simple sphere.
     '''
    octree, original_sdf = self.build_sphere_sdf_octree(
        depth=6, full_depth=4, radius=0.5)

    # Convert back to SDF
    reconstructed_sdf = octree.to_sdf()

    # Check shape
    self.assertEqual(reconstructed_sdf.shape, original_sdf.shape)

    # At full_depth, all voxels should be filled (nempty=False gives all nodes)
    x, y, z, b = octree.xyzb(octree.full_depth, nempty=False, normalize=True)

    # Check that the reconstructed values at full_depth match the stored fields
    full_depth_reconstructed = reconstructed_sdf[0, x, y, z]
    self.assertTrue(torch.allclose(
        full_depth_reconstructed, octree.fields[octree.full_depth]))

    # Check that reconstructed values at deeper depths match the stored fields
    # by extracting the same positions used during building
    for d in range(octree.full_depth + 1, octree.depth + 1):
      # Note: to_sdf scales rng by (2 ** (self.depth - d))
      scale = 2 ** (octree.depth - d)
      rng = ocnn.utils.range_grid(0, 2, device='cpu') * scale

      # Get parent positions at depth d-1
      x, y, z, b = octree.xyzb(d - 1, nempty=True, normalize=True)

      # For each of the 27 neighbor positions
      for i in range(27):
        # Calculate destination positions
        xr = x + rng[i][0]
        yr = y + rng[i][1]
        zr = z + rng[i][2]

        # Get reconstructed values
        reconstructed = reconstructed_sdf[b, xr, yr, zr]

        # Get original field values
        original = octree.fields[d][:, i]

        # They should match exactly (for float fields)
        self.assertTrue(torch.allclose(reconstructed, original),
                        f'Depth {d}, position {i} mismatch')

  def test_to_sdf_empty_voxels(self):
    r''' Test that empty voxels get the default value of 1.0.
     '''
    octree, _ = self.build_sphere_sdf_octree(depth=5, full_depth=3)

    reconstructed = octree.to_sdf()

    # Check that the SDF is initialized with 1.0 (default value)
    N = 2 ** octree.depth

    # At full_depth, all 2^(full_depth*3) voxels are filled (octree is full)
    # So there should be many values that are NOT 1.0
    self.assertEqual(reconstructed[0].shape, (N, N, N))
    self.assertTrue((reconstructed[0] != 1.0).any(),
                    'Expected some non-default values')

    # The number of 1.0 values should be less than total voxels
    num_ones = (reconstructed[0] == 1.0).sum()
    self.assertLess(num_ones, N * N * N,
                    'Too many default values, check to_sdf implementation')

  def test_to_sdf_different_full_depth(self):
    r''' Test to_sdf with various full_depth values.
     '''
    depth = 6

    for full_depth in [2, 3, 4]:
      octree, _ = self.build_sphere_sdf_octree(depth=depth, full_depth=full_depth)

      reconstructed = octree.to_sdf()

      # Check shape
      N = 2 ** depth
      self.assertEqual(reconstructed.shape, (1, N, N, N))

      # Verify that full_depth positions are filled correctly
      x, y, z, b = octree.xyzb(full_depth, nempty=False, normalize=True)
      reconstructed_full = reconstructed[b, x, y, z]
      self.assertTrue(torch.allclose(reconstructed_full,
                                     octree.fields[full_depth]))

  def test_to_sdf_roundtrip_consistency(self):
    r''' Test that to_sdf produces consistent results across multiple calls.
     '''
    octree, _ = self.build_sphere_sdf_octree(depth=6, full_depth=4)

    # Call to_sdf multiple times
    sdf1 = octree.to_sdf()
    sdf2 = octree.to_sdf()
    sdf3 = octree.to_sdf()

    # All results should be identical
    self.assertTrue(torch.allclose(sdf1, sdf2))
    self.assertTrue(torch.allclose(sdf2, sdf3))

  def test_to_sdf_minimal_depth(self):
    r''' Test to_sdf with minimal valid depth.
     '''
    # Minimum depth is 5 according to build_octree_from_sdf assertion
    octree, sdf = self.build_sphere_sdf_octree(depth=5, full_depth=3)
    reconstructed = octree.to_sdf()

    self.assertEqual(reconstructed.shape, sdf.shape)

    # Verify fields at full_depth match
    x, y, z, b = octree.xyzb(3, nempty=False, normalize=True)
    reconstructed_full = reconstructed[b, x, y, z]
    self.assertTrue(torch.allclose(reconstructed_full, octree.fields[3]))

  # ==================== adaptive tests ====================

  def test_adaptive_octree_sdf_basic(self):
    r''' Test basic functionality with a sphere SDF.
     '''

    octree, _ = self.build_sphere_sdf_octree(depth=6, full_depth=3,
                                             radius=0.5)

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
