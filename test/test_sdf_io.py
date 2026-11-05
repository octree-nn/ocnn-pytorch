"""Check SDF octree roundtrips, queries and batch counts."""

import tempfile
import unittest
from pathlib import Path

import torch

from ocnn.octree import Octree, octree_query_sdf
from ocnn.octree.sdf_io import load_sdf_octree, save_sdf_octree


class TestSdfIO(unittest.TestCase):

  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.filename = Path(self.directory.name) / 'shape.pt'
    axis = torch.linspace(-1, 1, 33)
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), dim=-1)
    self.tree = Octree(5, 3)
    self.sdf = (xyz.norm(dim=-1) - 0.3)[None]
    self.tree.build_octree_from_sdf(self.sdf)

  def test_roundtrip_queries_and_metadata(self):
    metadata = {'source': {'sample': '02691156/example', 'mtime_ns': 42}}
    size = save_sdf_octree(self.tree, self.filename, metadata)
    loaded, actual = load_sdf_octree(self.filename)
    self.assertEqual(size, self.filename.stat().st_size)
    self.assertEqual(actual, metadata)
    for name in ('keys', 'children', 'fields', 'nnum', 'nnum_nempty'):
      torch.testing.assert_close(
          getattr(loaded, name), getattr(self.tree, name), atol=0, rtol=0)
    generator = torch.Generator().manual_seed(17)
    points = torch.rand(4000, 4, generator=generator) * 32
    points[:, 3] = 0
    torch.testing.assert_close(octree_query_sdf(self.tree, points),
                               octree_query_sdf(loaded, points), equal_nan=True)
    self.assertTrue(torch.equal(loaded.batch_nnum[:, 0], loaded.nnum))
    torch.testing.assert_close(loaded.batch_nnum_nempty[:, 0],
                               loaded.nnum_nempty)

  def test_batched_roundtrip(self):
    # A sphere and a constant negative field have different topology and signs.
    tree = Octree(5, 3, batch_size=2)
    volume = torch.cat([self.sdf, -torch.ones_like(self.sdf)])
    tree.build_octree_from_sdf(volume)
    save_sdf_octree(tree, self.filename)
    loaded, _ = load_sdf_octree(self.filename)
    generator = torch.Generator().manual_seed(17)
    points = torch.rand(4000, 4, generator=generator) * 32
    points[:, 3] = torch.arange(points.shape[0]) % 2
    torch.testing.assert_close(octree_query_sdf(tree, points),
                               octree_query_sdf(loaded, points), equal_nan=True)
    torch.testing.assert_close(loaded.batch_nnum.sum(dim=1), loaded.nnum)
    torch.testing.assert_close(loaded.batch_nnum_nempty.sum(dim=1),
                               loaded.nnum_nempty)
    # The constant field has only full layers, with no splits at full_depth.
    self.assertEqual(loaded.batch_nnum[:, 1].tolist(), [1, 8, 64, 512, 0, 0])
    self.assertEqual(loaded.batch_nnum_nempty[:, 1].tolist(),
                     [1, 8, 64, 0, 0, 0])


if __name__ == '__main__':
  torch.set_num_threads(2)
  unittest.main()
