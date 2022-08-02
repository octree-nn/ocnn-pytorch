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
from .utils import get_batch_octree


class TesOctreeDWConv(unittest.TestCase):

  def test_dwconv_with_conv(self):

    depth = 4
    channel = 30
    max_buffer = int(2e4)
    octree = get_batch_octree()
    kernel_size = [[3, 3, 3], [3, 1, 1], [1, 3, 1], [1, 1, 3],
                   [2, 2, 2], [3, 3, 1], [1, 3, 3], [3, 1, 3]]

    for i in range(len(kernel_size)):
      for stride in [1, 2]:
        for nempty in [True, False]:
          nnum = octree.nnum_nempty[depth] if nempty else octree.nnum[depth]
          rnd_data = torch.randn(nnum, channel)
          dwdata = rnd_data.clone().requires_grad_()
          dwconv = ocnn.nn.OctreeDWConv(
              channel, kernel_size[i], stride, nempty, max_buffer=max_buffer)
          dwout = dwconv(dwdata, octree, depth)
          dwout.sum().backward()

          outs = []
          convs = []
          data = rnd_data.clone().requires_grad_()
          for c in range(channel):
            conv = ocnn.nn.OctreeConv(1, 1, kernel_size[i], stride, nempty)
            conv.weights.data.copy_(dwconv.weights.data[:, :, c:c+1])
            outs.append(conv(data[:, c:c+1], octree, depth))
            convs.append(conv)
          out = torch.cat(outs, dim=1)
          out.sum().backward()
          weight_grad = torch.cat([conv.weights.grad for conv in convs], dim=-1)

          self.assertTrue(torch.allclose(out, dwout, atol=1e-6))
          self.assertTrue(torch.allclose(data.grad, dwdata.grad, atol=1e-6))
          self.assertTrue(torch.allclose(weight_grad, dwconv.weights.grad,
                                         atol=5e-5))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
