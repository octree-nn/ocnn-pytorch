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

    linear = ocnn.nn.octree_trilinear_pts(
        data, octree, depth, pts, nempty=False)
    linear_ne = ocnn.nn.octree_trilinear_pts(
        data_ne, octree, depth, pts, nempty=True)
    near = ocnn.nn.octree_nearest_pts(
        data, octree, depth, pts, nempty=False)
    near_ne = ocnn.nn.octree_nearest_pts(
        data_ne, octree, depth, pts, nempty=True)

    self.assertTrue(np.allclose(linear.numpy(), test['linear'], atol=1.e-6))
    self.assertTrue(np.allclose(linear_ne.numpy(), test['linear_ne'], atol=1e-6))  # noqa
    self.assertTrue(np.allclose(near.numpy(), test['near'], atol=1e-6))
    self.assertTrue(np.allclose(near_ne.numpy(), test['near_ne'], atol=1e-6))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
