# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import unittest
from unittest.mock import patch

import torch

import ocnn
from ocnn.octree.adaptive import adaptive_octree_sdf
from ocnn.octree.sdf import _query_corner_sign


class TestSdfOctree(unittest.TestCase):

  def build_sphere_sdf_octree(self, depth=6, full_depth=4, radius=0.5,
                              compress=False, device='cpu'):
    r''' Build an octree from a sphere SDF.
     '''
    resolution = 2 ** depth + 1
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
    resolution = 2 ** depth + 1
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
    octree, original = self.build_sphere_sdf_octree(
        depth=6, full_depth=2, radius=0.6)
    reconstructed = octree.to_sdf()
    self.assertEqual(reconstructed[0, 33, 32, 32].item(), -1.0)
    self.assertTrue(torch.equal(reconstructed > 0, original > 0))

  def test_to_sdf_signs_and_surface_corners(self):
    # Include a cavity, a plane crossing the domain boundary, and empty trees.
    axis = torch.linspace(-1, 1, 33)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
    radius = torch.sqrt(x*x + y*y + z*z)
    original = torch.stack([(radius - 0.5).abs() - 0.15,
                            x + y + z - 0.2,
                            torch.ones_like(x), -torch.ones_like(x)])
    for compress in (False, True):
      with self.subTest(compress=compress):
        octree = ocnn.octree.Octree(depth=5, full_depth=2, batch_size=4)
        octree.build_octree_from_sdf(original, compress=compress)
        reference = original
        if compress:
          reference = ((original * octree.field_scale).round()
                       .clamp(-32768, 32767) / octree.field_scale)
        reconstructed = octree.to_sdf()
        self.assertEqual(reconstructed.shape, original.shape)
        self.assertTrue(torch.equal(reconstructed > 0, reference > 0))

        # Independently identify every surface cell and verify all its corners.
        positive = reference > 0
        origin = positive[:, :-1, :-1, :-1]
        offsets = [(i, j, k) for i in (0, 1)
                   for j in (0, 1) for k in (0, 1)]
        mixed = torch.zeros_like(origin)
        for i, j, k in offsets:
          mixed |= origin != positive[:, i:i+32, j:j+32, k:k+32]
        for i, j, k in offsets:
          actual = reconstructed[:, i:i+32, j:j+32, k:k+32][mixed]
          expected = reference[:, i:i+32, j:j+32, k:k+32][mixed]
          self.assertTrue(torch.equal(actual, expected))

    for value in (-0.25, 0.25):
      octree = ocnn.octree.Octree(depth=5, full_depth=2)
      octree.build_octree_from_sdf(torch.full((1, 33, 33, 33), value))
      self.assertTrue((octree.to_sdf() > 0).eq(value > 0).all())

  def test_to_sdf_different_full_depth(self):
    r''' Test to_sdf with various full_depth values.
     '''
    depth = 6

    for full_depth in [2, 3, 4]:
      octree, _ = self.build_sphere_sdf_octree(depth=depth, full_depth=full_depth)

      reconstructed = octree.to_sdf()

      # Check shape
      N = 2 ** depth + 1
      self.assertEqual(reconstructed.shape, (1, N, N, N))

      # Verify that full_depth positions are filled correctly
      x, y, z, b = octree.xyzb(full_depth, nempty=False, normalize=True)
      reconstructed_full = reconstructed[b, x, y, z]
      self.assertTrue(torch.allclose(reconstructed_full,
                                     octree.fields[full_depth]))

  def test_to_sdf_saved_samples_and_sign_only_blocks(self):
    self.check_saved_samples_and_sign_only_blocks('cpu')

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
  def test_to_sdf_saved_samples_and_sign_only_blocks_cuda(self):
    self.check_saved_samples_and_sign_only_blocks('cuda')

  def check_saved_samples_and_sign_only_blocks(self, device):
    # A shell creates interior/exterior terminal blocks and coarse/fine faces;
    # the plane exercises upper faces, edges and corners with both signs.
    axis = torch.linspace(-1, 1, 33, device=device)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
    radius = (x*x + y*y + z*z).sqrt()
    source = torch.stack([(radius - 0.55).abs() - 0.12,
                          x + y - z - 0.1, torch.zeros_like(x)])
    for full_depth in (0, 2, 4):
      for compress in (False, True):
        with self.subTest(full_depth=full_depth, compress=compress):
          tree = ocnn.octree.Octree(5, full_depth, batch_size=3, device=device)
          tree.build_octree_from_sdf(source, compress=compress)
          reference = source
          if compress:
            reference = ((source * tree.field_scale).round()
                         .clamp(-32768, 32767) / tree.field_scale)
          saved = torch.zeros_like(source, dtype=torch.bool)
          cx, cy, cz, b = tree.xyzb(full_depth, normalize=True)
          saved[b, cx, cy, cz] = True
          for d in range(full_depth + 1, 6):
            cx, cy, cz, b = tree.xyzb(d - 1, nempty=True, normalize=True)
            for dx, dy, dz in (ocnn.utils.range_grid(0, 2, device=device) *
                               (2 ** (5-d))):
              saved[b, cx+dx, cy+dy, cz+dz] = True
          expected = torch.where(reference > 0, 1.0, -1.0)
          expected[saved] = reference[saved]
          self.assertTrue(torch.equal(tree.to_sdf(), expected))

  def check_corner_queries(self, device):
    axis = torch.arange(33, device=device)
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), dim=-1)
    plane = (xyz[..., 0] + xyz[..., 1] - xyz[..., 2] - 16).float() / 64
    isolated = -torch.ones_like(plane)
    isolated[-1, -1, -1] = 0.25
    original = torch.stack([plane, isolated, torch.zeros_like(plane)])
    points = xyz.reshape(-1, 3).repeat(3, 1)
    batch = torch.arange(3, device=device).repeat_interleave(33 ** 3)
    for full_depth in (0, 2, 4):
      for compress in (False, True):
        with self.subTest(device=device, full_depth=full_depth,
                          compress=compress):
          octree = ocnn.octree.Octree(5, full_depth, batch_size=3, device=device)
          octree.build_octree_from_sdf(original, compress=compress)
          signs = _query_corner_sign(octree, points, batch)
          self.assertTrue(torch.equal(signs > 0, original.flatten() > 0))
          # Reconstruction must never invoke the arbitrary-point interpolator.
          with patch('ocnn.octree.sdf._interpolate',
                     side_effect=AssertionError('Unexpected interpolation')):
            restored = octree.to_sdf()
          self.assertTrue(torch.equal(restored > 0, original > 0))
          self.assertEqual(restored[1, -1, -1, -1].item(), 0.25)

  def test_to_sdf_corner_queries(self):
    self.check_corner_queries('cpu')

  def test_to_sdf_multiple_chunks(self):
    generator = torch.Generator().manual_seed(42)
    original = torch.rand((9, 33, 33, 33), generator=generator) - 0.5
    octree = ocnn.octree.Octree(5, 1, batch_size=9)
    octree.build_octree_from_sdf(original)
    restored = octree.to_sdf()
    self.assertTrue(torch.equal(restored > 0, original > 0))
    # Random signs allocate essentially the whole tree; verify every stored
    # sample independently, including batch and chunk boundaries.
    for d in range(2, 6):
      x, y, z, b = octree.xyzb(d - 1, nempty=True, normalize=True)
      offsets = ocnn.utils.range_grid(0, 2) * (2 ** (5 - d))
      for dx, dy, dz in offsets:
        self.assertTrue(torch.equal(restored[b, x+dx, y+dy, z+dz],
                                    original[b, x+dx, y+dy, z+dz]))

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
  def test_to_sdf_corner_queries_cuda(self):
    self.check_corner_queries('cuda')

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
