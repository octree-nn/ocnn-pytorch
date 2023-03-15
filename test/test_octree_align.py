# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import unittest

import ocnn
from .utils import get_octree


class TestOctreeAlign(unittest.TestCase):

  def test_octree_align(self):
    octree1 = get_octree(1)
    octree2 = get_octree(2)

    depth = 4
    octree = ocnn.octree.merge_octrees([octree1, octree2])

    # nempty = False
    num = octree.nnum[depth]
    num1 = octree1.nnum[depth]
    value = torch.rand(num, 3)
    gt = value[:num1]
    out = ocnn.nn.octree_align(value, octree, octree1, depth, nempty=False)
    self.assertTrue(torch.equal(out, gt))

    # nempty = True
    num = octree.nnum_nempty[depth]
    num1 = octree1.nnum_nempty[depth]
    value = torch.rand(num, 3)
    gt = value[:num1]
    out = ocnn.nn.octree_align(value, octree, octree1, depth, nempty=True)
    self.assertTrue(torch.equal(out, gt))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
