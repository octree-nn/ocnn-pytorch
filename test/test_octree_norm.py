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


class OctreeNormTest(unittest.TestCase):

  def test_instance_norm(self):
    octree1 = get_octree(4)
    octree2 = get_octree(5)
    octree = ocnn.octree.merge_octrees([octree1, octree2])

    # test 1 - full layer for depth 2
    depth = 2
    channel = 7
    nnum = octree1.nnum[depth]
    data = torch.rand(2, channel, nnum)

    instance_norm = torch.nn.InstanceNorm1d(channel)
    out_gt = instance_norm(data)

    octree_norm = ocnn.nn.OctreeInstanceNorm(channel, nempty=False)
    data = data.permute(0, 2, 1).reshape(-1, channel)
    out = octree_norm(data, octree, depth)
    out = out.view(2, nnum, channel).permute(0, 2, 1)

    self.assertTrue(torch.allclose(out, out_gt, atol=5e-6))

    # test 2
    for nempty in [True, False]:
      depth = 4
      nnum1 = octree1.nnum_nempty[depth] if nempty else octree1.nnum[depth]
      data1 = torch.rand(1, channel, nnum1)
      out_gt1 = instance_norm(data1)

      nnum2 = octree2.nnum_nempty[depth] if nempty else octree2.nnum[depth]
      data2 = torch.rand(1, channel, nnum2)
      out_gt2 = instance_norm(data2)
      out_gt = torch.cat([out_gt1, out_gt2], dim=2).squeeze(0).transpose(0, 1)

      data = torch.cat([data1, data2], dim=2).squeeze(0).transpose(0, 1)
      octree_norm = ocnn.nn.OctreeInstanceNorm(channel, nempty)
      out = octree_norm(data, octree, depth)

      self.assertTrue(torch.allclose(out, out_gt, atol=5e-6))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
