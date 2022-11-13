# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import numpy as np
import unittest

import ocnn
from .utils import get_batch_octree, get_octree


class TesOctreePool(unittest.TestCase):

  def test_octree_max_pool(self):
    r''' Tests octree_max_pool/octree_max_unpool. '''

    folder = os.path.dirname(__file__)
    data = np.load(os.path.join(folder, 'data/octree_nn.npz'))
    octree = get_batch_octree()

    depth = data['depth'].item()
    data_in = [torch.from_numpy(data['data_0']),
               torch.from_numpy(data['data_1'])]

    pool, idx = ocnn.nn.octree_max_pool(
        data_in[0], octree, depth, return_indices=True)
    self.assertTrue(np.array_equal(pool.numpy(), data['pool']))
    upool = ocnn.nn.octree_max_unpool(pool, idx, octree, depth-1)
    self.assertTrue(np.array_equal(upool.numpy(), data['upool']))

  def test_octree_mean_pool(self):
    r''' Tests octree_mean_pool. '''

    depth = 3
    channel = 33
    octree = get_batch_octree()  # full_depth: 3
    num = octree.nnum[depth]
    data = torch.rand(num, channel)
    data_gt = ocnn.nn.octree2voxel(data, octree, depth, nempty=False)

    for kernel, stride, pad in [(3, 1, 1), (3, 2, 1), (2, 2, 0)]:
      pool3d = torch.nn.AvgPool3d(
          kernel_size=kernel, stride=stride, padding=pad,
          count_include_pad=False)
      out_gt = pool3d(data_gt.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

      octree_pool = ocnn.nn.OctreeAvgPool(
          kernel_size=[kernel], stride=stride, nempty=False)
      out = octree_pool(data, octree, depth)
      curr_depth = depth - 1 if stride == 2 else depth
      out = ocnn.nn.octree2voxel(out, octree, curr_depth, nempty=False)
      self.assertTrue(torch.allclose(out, out_gt))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
