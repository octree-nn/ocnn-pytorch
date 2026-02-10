# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang, Chuan-Zhi Zhou
# --------------------------------------------------------

import os
import torch
import unittest

import ocnn
import ocnn.nn.kernels.config
from ocnn.octree import Points, Octree
from ocnn.models import ResNet

from .utils import sphere_coords, skip_triton_test


# !!! disable TF32 for testing
ocnn.nn.kernels.config.allow_tf32 = False
# !!! disable triton for octree_conv
ocnn.nn.octree_conv.DISABLE_TRITON = True


@unittest.skipIf(skip_triton_test(), "Skip triton")
class TestConvertConvTriton(unittest.TestCase):

  def test_resnet(self):
    atol = 5e-3
    octree = self.build_octree()
    octree = ocnn.octree.merge_octrees([octree, octree])
    octree.construct_all_neigh()
    depth = octree.depth
    depth2channel = {3: 1024, 4: 512, 5: 256, 6: 128, 7: 64, 8: 32}
    in_channel = depth2channel[depth]
    out_channel = 1000
    stages = 4
    nempty = False
    num = octree.nnum[depth] if not nempty else octree.nnum_nempty[depth]
    dtype = torch.float32

    model = ResNet(in_channel, out_channel, resblock_num=2,
                   stages=stages, nempty=nempty, dropout=0.0)  # no dropout
    model = model.cuda()
    data = torch.randn(num, in_channel, device='cuda', dtype=dtype)
    out = model(data, octree, depth)

    model_triton = ocnn.nn.convert_conv_triton(model)
    self.assertTrue(isinstance(model_triton, ResNet))
    out_triton = model_triton(data, octree, depth)
    self.assertTrue(torch.allclose(out, out_triton, atol=atol))

  def build_octree(self):
    r = 64
    depth, full_depth = 6, 3
    pos = sphere_coords(64, device='cuda')
    pos = pos / r * 2.0 - 1.0  # normalize to [-1,1]
    points = Points(points=pos)
    octree = Octree(depth, full_depth, device='cuda')
    octree.build_octree(points)
    # octree.construct_all_neigh()
    return octree


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
