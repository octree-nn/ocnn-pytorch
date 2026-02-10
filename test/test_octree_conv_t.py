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
from .utils import sphere_coords, skip_triton_test

# !!! disable TF32 for testing !!!
ocnn.nn.kernels.config.allow_tf32 = False


@unittest.skipIf(skip_triton_test(), "Skip triton")
class TestOctreeConvTriton(unittest.TestCase):

  def test_conv(self):
    octree = self.build_octree()
    depth2channel = {3: 1024, 4: 512, 5: 256, 6: 128, 7: 64}
    for d in [octree.depth, octree.depth - 1]:
      for out_ratio in [1.0, 0.5, 2.0]:
        self.conv_forward_backward(d, out_ratio, octree, depth2channel[d])

  def test_conv_small_channel(self):
    octree = self.build_octree()
    for d in [octree.depth]:
      for out_ratio in [1.0, 2.0]:
        for channel in [4, 8, 16]:
          # print(f'Testing depth={d}, out_ratio={out_ratio}, channel={channel}')
          self.conv_forward_backward(d, out_ratio, octree, channel)

  def test_conv_irregular_channel(self):
    octree = self.build_octree()
    for d in [octree.depth]:
      for out_ratio in [1.0, 2.0]:
        for channel in [2 ** i + 5 for i in range(6, 8)]:
          # print(f'Testing depth={d}, out_ratio={out_ratio}, channel={channel}')
          self.conv_forward_backward(d, out_ratio, octree, channel)

  def build_octree(self):
    r = 64
    depth, full_depth = 7, 3
    pos = sphere_coords(64, device='cuda')
    pos = pos / r * 2.0 - 1.0  # normalize to [-1,1]
    points = Points(points=pos)
    octree = Octree(depth, full_depth, device='cuda')
    octree.build_octree(points)
    octree.construct_all_neigh()
    return octree

  def conv_forward_backward(self, depth, out_ratio, octree, in_channel):
    atol = 5e-3
    kernel_size = [3, 3, 3]
    nempty = False
    device = 'cuda'
    out_channel = int(in_channel * out_ratio)

    # initialize conv layers
    conv_pt = ocnn.nn.OctreeConv(
        in_channel, out_channel, kernel_size, stride=1,
        nempty=nempty, use_bias=True, method='block_gemm').to(device)
    conv_tt = ocnn.nn.OctreeConvTriton(
        in_channel, out_channel, kernel_size, stride=1,
        nempty=nempty, use_bias=True,).to(device)
    with torch.no_grad():
      conv_tt.weights.copy_(conv_pt.weights)
      conv_tt.bias.copy_(conv_pt.bias)

    # initialize data and grad
    data = torch.randn(octree.nnum[depth], in_channel, device=device)
    data_tt = data.detach().clone().requires_grad_()
    data_pt = data.detach().clone().requires_grad_()
    grad = torch.randn(octree.nnum[depth], out_channel, device=device)

    # forward
    out_tt = conv_tt(data_tt, octree, depth)
    out_pt = conv_pt(data_pt, octree, depth)

    # backward
    loss_tt = (out_tt * grad).sum()
    loss_pt = (out_pt * grad).sum()
    loss_tt.backward()
    loss_pt.backward()

    # check results
    msg = f'depth: {depth}, out_ratio: {out_ratio}'
    self.assertTrue(torch.allclose(out_tt, out_pt, atol=atol), msg)
    self.assertTrue(torch.allclose(data_pt.grad, data_tt.grad, atol=atol), msg)
    # TODO: depth: 7, out_ratio: 2.0, error: 0.0031270573381334543
    err = f', error: {self.calc_err(conv_pt.weights.grad, conv_tt.weights.grad)}'
    self.assertTrue(torch.allclose(
        conv_pt.weights.grad, conv_tt.weights.grad, atol=1e-2), msg + err)
    self.assertTrue(torch.allclose(
        conv_pt.bias.grad, conv_tt.bias.grad, atol=atol), msg)

  def calc_err(self, src, ref):
    abs_err = (src - ref).float().abs()
    return abs_err.max().item()  # , err.mean().item()


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
