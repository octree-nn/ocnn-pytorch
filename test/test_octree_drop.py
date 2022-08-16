# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import ocnn
import unittest

from .utils import get_octree


class OctreeDropTest(unittest.TestCase):

  def test_octree_drop_path(self):
    r'''Just execute the `OctreeDropPath`, and there are no comparisons with 
    ground-truth results.
    '''

    octrees = [get_octree(i) for i in ([4] * 8 + [5] * 2)]
    octree = ocnn.octree.merge_octrees(octrees)

    # Test 1
    depth = 5
    nnum = octree.nnum[depth]
    data = torch.rand(nnum, 3)
    drop_path = ocnn.nn.OctreeDropPath(drop_prob=0.8, nempty=False)
    output = drop_path(data, octree, depth)

    # Test 2
    nnum_nempty = octree.nnum_nempty[depth]
    data = torch.rand(nnum_nempty, 3)
    drop_path = ocnn.nn.OctreeDropPath(drop_prob=0.8, nempty=True)
    output = drop_path(data, octree, depth)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
