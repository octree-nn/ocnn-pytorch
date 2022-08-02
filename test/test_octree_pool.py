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

  def test_octree_pool(self):
    r''' Tests octree_max_pool/octree_max_unpool.
    '''

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


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
