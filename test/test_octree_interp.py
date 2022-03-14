import os
import torch
import numpy as np
import unittest

import ocnn
from .utils import get_batch_octree


class TesOctreeInterp(unittest.TestCase):

  def test_octree_interp(self):

    folder = os.path.dirname(__file__)
    test = np.load(os.path.join(folder, 'data/interp.npz'))
    octree = get_batch_octree()

    depth = 5
    data = torch.from_numpy(test['data'])
    pts = torch.from_numpy(test['pts'])
    data_ne = ocnn.nn.octree_depad(data, octree, depth)

    linear = ocnn.nn.octree_linear_pts(
        data, octree, depth, pts, nempty=False)
    linear_ne = ocnn.nn.octree_linear_pts(
        data_ne, octree, depth, pts, nempty=True)
    near = ocnn.nn.octree_nearest_pts(
        data, octree, depth, pts, nempty=False)
    near_ne = ocnn.nn.octree_nearest_pts(
        data_ne, octree, depth, pts, nempty=True)

    self.assertTrue(np.allclose(linear.numpy(), test['linear'], atol=1.e-6))
    self.assertTrue(np.allclose(linear_ne.numpy(), test['linear_ne'], atol=1e-6))  # noqa
    self.assertTrue(np.allclose(near.numpy(), test['near'], atol=1e-6))
    self.assertTrue(np.allclose(near_ne.numpy(), test['near_ne'], atol=1e-6))

  def test_octree_upsample(self):

    depth = 4
    depth_out = 5
    octree = get_batch_octree()

    # test case: nempty=False
    nnum = octree.nnum[depth]
    data = torch.rand(nnum, 4)
    out = ocnn.nn.octree_upsample(data, octree, depth=depth, nempty=False)

    key = octree.keys[depth_out]
    xyzb = ocnn.octree.key2xyz(key, depth_out)
    pts = torch.stack(xyzb, dim=1)
    pts[:, :3] = pts[:, :3] * 0.5
    out_ref = ocnn.nn.octree_nearest_pts(
        data, octree, depth, pts, nempty=False, bound_check=True)

    self.assertTrue(np.array_equal(out.numpy(), out_ref.numpy()))

    # test case: nempty=False
    nnum = octree.nnum_nempty[depth]
    data = torch.rand(nnum, 4)
    out = ocnn.nn.octree_upsample(data, octree, depth=depth, nempty=True)

    key = octree.keys[depth_out]
    key = ocnn.nn.octree_depad(key, octree, depth_out)
    xyzb = ocnn.octree.key2xyz(key, depth_out)
    pts = torch.stack(xyzb, dim=1)
    pts[:, :3] = pts[:, :3] * 0.5
    out_ref = ocnn.nn.octree_nearest_pts(
        data, octree, depth, pts, nempty=True, bound_check=True)

    self.assertTrue(np.array_equal(out.numpy(), out_ref.numpy()))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
