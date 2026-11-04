import unittest

import torch
import torch.nn.functional as F

from ocnn.octree import Octree, octree_query_sdf


def dense_query(volume, pts):
  # grid_sample's coordinate order is (W, H, D), i.e. (z, y, x) here.
  grid = (pts[:, [2, 1, 0]] * (2.0 / (volume.shape[-1] - 1)) - 1)
  grid = grid[None, :, None, None, :].expand(volume.shape[0], -1, -1, -1, -1)
  sampled = F.grid_sample(volume[:, None], grid, mode='bilinear',
                          align_corners=True)[:, 0, :, 0, 0]
  return sampled[pts[:, 3].long(), torch.arange(pts.shape[0], device=pts.device)]


def sphere_volumes(depth, device='cpu'):
  axis = torch.linspace(-1, 1, 2 ** depth, device=device)
  xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), dim=-1)
  # The second object has an enclosed cavity, so an origin/coarse sign alone
  # cannot classify every point in an active region.
  sphere = xyz.norm(dim=-1) - 0.35
  shell = ((xyz - xyz.new_tensor([0.06, -0.04, 0.02])).norm(dim=-1) - 0.26).abs()
  shell = shell - 0.06
  return torch.stack([sphere, shell])


class TestSdfQuery(unittest.TestCase):

  def compare_dense(self, device, compiled=False):
    query = (torch.compile(octree_query_sdf, fullgraph=True)
             if compiled else octree_query_sdf)
    torch.manual_seed(42)
    depth = 6
    volume = sphere_volumes(depth, device)
    n = volume.shape[-1]
    pts = torch.rand(40000, 4, device=device)
    pts[:, :3] *= n - 1
    pts[:, 3] = torch.arange(pts.shape[0], device=device) % 2
    # Exact sample positions and the closed-domain endpoints.
    integer = torch.randint(n, (2000, 4), device=device).float()
    integer[:, 3] %= 2
    endpoints = pts.new_tensor([[0, 0, 0, 0], [n-1, n-1, n-1, 1],
                                [n-1, 0, n-1, 0], [31.5, 31.5, 31.5, 1]])
    pts = torch.cat([pts, integer, endpoints])
    reference = dense_query(volume, pts)
    for full_depth in (2, 3, 4):
      with self.subTest(device=device, full_depth=full_depth):
        octree = Octree(depth, full_depth, batch_size=2, device=device)
        octree.build_octree_from_sdf(volume)
        result = query(octree, pts)
        tensor_result = octree_query_sdf(octree, pts)
        torch.testing.assert_close(result, tensor_result, equal_nan=True)
        self.assertTrue(result.valid.all())
        self.assertTrue(result.has_sdf.any())
        self.assertTrue((~result.has_sdf).any())
        cells = pts.clone()
        cells[:, :3] = pts[:, :3].floor().clamp(max=n - 2)
        # Binary-search membership independently checks the gather traversal.
        expected_coverage = octree.search_xyzb(cells, depth, nempty=False) >= 0
        self.assertTrue(torch.equal(result.has_sdf, expected_coverage))
        self.assertTrue(torch.isnan(result.sdf[~result.has_sdf]).all())
        torch.testing.assert_close(result.sdf[result.has_sdf],
                                   reference[result.has_sdf], atol=2.e-6, rtol=0)
        self.assertTrue(torch.equal(result.sign > 0, reference > 0))

  def test_dense_agreement_cpu(self):
    self.compare_dense('cpu')

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
  def test_dense_agreement_cuda(self):
    self.compare_dense('cuda', compiled=True)

  def test_constant_fields_and_empty_queries(self):
    for value in (-0.2, 0.0, 0.2):
      volume = torch.full((2, 32, 32, 32), value)
      octree = Octree(5, 2, batch_size=2)
      octree.build_octree_from_sdf(volume)
      pts = torch.tensor([[0., 0., 0., 0.], [31., 31., 31., 1.]])
      result = octree_query_sdf(octree, pts)
      self.assertTrue(result.valid.all())
      self.assertFalse(result.has_sdf.any())
      self.assertTrue((result.sign == (1 if value > 0 else -1)).all())
      empty = octree_query_sdf(octree, torch.empty(0, 4))
      self.assertEqual(empty.sdf.shape, (0,))

  def test_invalid_points(self):
    octree = Octree(5, 2)
    octree.build_octree_from_sdf(torch.ones(1, 32, 32, 32))
    pts = torch.tensor([[-0.01, 0, 0, 0], [31.01, 0, 0, 0],
                        [0, 0, 0, -1], [0, 0, 0, 1], [0, 0, 0, 0.5],
                        [float('nan'), 0, 0, 0], [0, float('inf'), 0, 0]])
    result = octree_query_sdf(octree, pts)
    self.assertFalse(result.valid.any())
    self.assertFalse(result.has_sdf.any())
    self.assertTrue(torch.isnan(result.sdf).all())
    self.assertTrue((result.sign == 0).all())

  def test_point_gradients_match_dense(self):
    volume = sphere_volumes(5)[:1].double()
    octree = Octree(5, 3)
    octree.build_octree_from_sdf(volume)
    pts = torch.tensor([[20.2, 16.1, 15.7, 0.],
                        [15.6, 20.3, 16.2, 0.]], dtype=torch.double,
                       requires_grad=True)
    result = octree_query_sdf(octree, pts)
    self.assertTrue(result.has_sdf.all())
    actual = torch.autograd.grad(result.sdf.sum(), pts)[0]
    expected = torch.autograd.grad(dense_query(volume, pts).sum(), pts)[0]
    torch.testing.assert_close(actual, expected, atol=1.e-12, rtol=1.e-10)

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
  def test_compiled_cuda_point_gradients(self):
    volume = sphere_volumes(5, 'cuda')[:1]
    octree = Octree(5, 3, device='cuda')
    octree.build_octree_from_sdf(volume)
    pts = torch.tensor([[20.2, 16.1, 15.7, 0.],
                        [15.6, 20.3, 16.2, 0.]], device='cuda',
                       requires_grad=True)
    query = torch.compile(octree_query_sdf, fullgraph=True)
    result = query(octree, pts)
    actual = torch.autograd.grad(result.sdf.sum(), pts)[0]
    expected = torch.autograd.grad(dense_query(volume, pts).sum(), pts)[0]
    torch.testing.assert_close(actual, expected, atol=1.e-6, rtol=1.e-5)

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
  def test_compiled_cuda_constant_invalid_and_strided_queries(self):
    query = torch.compile(octree_query_sdf, fullgraph=True)
    for value in (-0.2, 0.0, 0.2):
      octree = Octree(5, 2, batch_size=2, device='cuda')
      octree.build_octree_from_sdf(torch.full((2, 32, 32, 32), value,
                                             device='cuda'))
      pts = torch.tensor([[0., 0., 0., 0.], [31., 31., 31., 1.],
                          [-1., 0., 0., 0.], [0., 0., 0., 0.5],
                          [float('nan'), 0., 0., 0.]], device='cuda')
      storage = torch.zeros(5, 8, device='cuda')
      storage[:, ::2] = pts
      pts = storage[:, ::2]
      actual = query(octree, pts)
      expected = octree_query_sdf(octree, pts)
      torch.testing.assert_close(actual, expected, equal_nan=True)
      empty = query(octree, pts[:0])
      self.assertEqual(empty.sdf.shape, (0,))


if __name__ == '__main__':
  unittest.main()
