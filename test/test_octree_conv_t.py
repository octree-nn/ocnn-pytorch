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
from ocnn.octree import Points, Octree
from .utils import sphere_coords


class TesOctreeConv(unittest.TestCase):

  def test_conv_forward(self):
    r''' Tests octree2col/col2octree, octree_conv/octree_deconv.
    '''

    r = 64
    pos = sphere_coords(64, device='cuda')
    pos = pos / r * 2.0 - 1.0  # normalize to [-1,1]
    points = Points(points=pos)
    octree = Octree(depth=7, full_depth=2, device='cuda')
    octree.build_octree(points)
    octree.construct_all_neigh()

    ci, co = 32, 16
    num = octree.nnum[7]
    data = torch.randn(num, ci, device='cuda')
    conv = ocnn.nn.OctreeConv(
        ci, co, kernel_size=[3, 3, 3], stride=1, use_bias=True).cuda()
    conv_t = ocnn.nn.OctreeConvTriton(
        ci, co, kernel_size=[3, 3, 3], stride=1, use_bias=True).cuda()
    conv_t.weights.data.copy_(conv.weights.data)
    conv_t.bias.data.copy_(conv.bias.data)

    out = conv(data, octree, depth=7)
    out_t = conv_t(data, octree, depth=7)
    self.assertTrue(torch.allclose(out, out_t, atol=1e-6))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
