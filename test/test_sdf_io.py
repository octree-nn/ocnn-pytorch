"""Check on-disk fidelity, format rejection and failed-write isolation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ocnn.octree import Octree, octree_query_sdf
from ocnn.octree.sdf_io import (
  QUERY_DOMAIN,
  VERSION,
  load_sdf_octree,
  save_sdf_octree,
)


class TestSdfIO(unittest.TestCase):

  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.filename = Path(self.directory.name) / 'nested/shape.pt'
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
    payload = torch.load(self.filename, weights_only=True)
    self.assertEqual(payload['query_domain'], QUERY_DOMAIN)
    self.assertEqual(payload['query_upper_bound'], 32)
    self.assertNotIn('valid_coordinate_max', payload)
    upper = torch.tensor([[32., 0., 0., 0.], [0., 32., 0., 0.],
                           [0., 0., 32., 0.]])
    self.assertFalse(octree_query_sdf(loaded, upper).valid.any())

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

  def test_invalid_fields_do_not_replace_existing_output(self):
    save_sdf_octree(self.tree, self.filename)
    original = self.filename.read_bytes()
    for dtype, compress in ((torch.float64, False), (torch.float32, True)):
      with self.subTest(dtype=dtype, compress=compress):
        tree = Octree(5, 3)
        tree.build_octree_from_sdf(self.sdf.to(dtype), compress=compress)
        with self.assertRaisesRegex(ValueError, 'float32 SDF field'):
          save_sdf_octree(tree, self.filename)
        self.assertEqual(self.filename.read_bytes(), original)
        self.assertEqual(list(self.filename.parent.glob('*.tmp')), [])

  def test_reject_legacy_closed_domain_and_wrong_query_metadata(self):
    save_sdf_octree(self.tree, self.filename)
    payload = torch.load(self.filename, weights_only=True)
    payload['version'] = 2
    torch.save(payload, self.filename)
    with self.assertRaisesRegex(ValueError, 'version'):
      load_sdf_octree(self.filename)
    payload['version'] = VERSION
    payload['query_domain'] = 'closed'
    torch.save(payload, self.filename)
    with self.assertRaisesRegex(ValueError, 'coordinate metadata'):
      load_sdf_octree(self.filename)

  def test_reject_corrupt_field_shape(self):
    save_sdf_octree(self.tree, self.filename)
    payload = torch.load(self.filename, weights_only=True)
    payload['fields'][5] = payload['fields'][5][:, :26]
    torch.save(payload, self.filename)
    with self.assertRaisesRegex(ValueError, 'field'):
      load_sdf_octree(self.filename)

  def test_interrupted_write_preserves_existing_output(self):
    def fail(payload, filename):
      Path(filename).write_bytes(b'incomplete')
      raise OSError('Simulated interrupted write')

    for existing in (False, True):
      with self.subTest(existing=existing):
        if existing:
          save_sdf_octree(self.tree, self.filename)
          original = self.filename.read_bytes()
        with (patch('ocnn.octree.sdf_io.torch.save', side_effect=fail),
              self.assertRaisesRegex(OSError, 'interrupted')):
          save_sdf_octree(self.tree, self.filename)
        if existing:
          self.assertEqual(self.filename.read_bytes(), original)
        else:
          self.assertFalse(self.filename.exists())
        self.assertEqual(list(self.filename.parent.glob('*.tmp')), [])


if __name__ == '__main__':
  torch.set_num_threads(2)
  unittest.main()
