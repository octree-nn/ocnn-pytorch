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


class TesOctreeConv(unittest.TestCase):

  def test_deconv_specialized(self):
    octree = get_batch_octree()
    depth = 4
    in_channels = 3
    out_channels = 4

    for nempty in [False, True]:
      nnum = octree.nnum_nempty[depth] if nempty else octree.nnum[depth]
      data = torch.randn(nnum, in_channels).t().contiguous().t()
      self.assertFalse(data.is_contiguous())

      deconv_ref = ocnn.nn.OctreeDeconv(
          in_channels, out_channels, [2, 2, 2], 2, nempty=nempty,
          method='explicit_gemm', use_bias=True)
      deconv = ocnn.nn.OctreeDeconvK2(
          in_channels, out_channels, nempty=nempty, use_bias=True)
      deconv.weights.data.copy_(deconv_ref.weights.data)
      deconv_ref.bias.data.fill_(0.25)
      deconv.bias.data.copy_(deconv_ref.bias.data)

      data_ref = data.detach().clone().requires_grad_()
      data = data.detach().clone().requires_grad_()
      out_ref = deconv_ref(data_ref, octree, depth)
      out = deconv(data, octree, depth)
      self.assertTrue(torch.allclose(out, out_ref, atol=1.0e-6))

      out_ref.sum().backward()
      out.sum().backward()
      self.assertTrue(torch.allclose(data.grad, data_ref.grad, atol=1.0e-6))
      self.assertTrue(torch.allclose(
          deconv.weights.grad, deconv_ref.weights.grad, atol=1.0e-6))
      self.assertTrue(torch.allclose(
          deconv.bias.grad, deconv_ref.bias.grad, atol=1.0e-6))

  def test_conv_specialized(self):
    octree = get_batch_octree()
    depth = 5
    in_channels = 3
    out_channels = 4

    for nempty in [False, True]:
      nnum = octree.nnum_nempty[depth] if nempty else octree.nnum[depth]
      data = torch.randn(nnum, in_channels)
      if not nempty:
        data = data.t().contiguous().t()
        self.assertFalse(data.is_contiguous())

      conv_ref = ocnn.nn.OctreeConv(
          in_channels, out_channels, [2, 2, 2], 2, nempty=nempty,
          method='explicit_gemm', use_bias=True)
      conv = ocnn.nn.OctreeConvK2(
          in_channels, out_channels, nempty=nempty, use_bias=True)
      conv.weights.data.copy_(conv_ref.weights.data)
      conv_ref.bias.data.fill_(0.25)
      conv.bias.data.copy_(conv_ref.bias.data)

      out_ref = conv_ref(data, octree, depth)
      out = conv(data, octree, depth)
      self.assertTrue(torch.allclose(out, out_ref, atol=1.0e-6))

    data = torch.randn(octree.nnum[depth], in_channels)
    conv = ocnn.nn.OctreeConvK1(
        in_channels, out_channels, use_bias=True)

    out_ref = data @ conv.linear.weight.t() + conv.linear.bias
    out = conv(data)
    self.assertTrue(torch.allclose(out, out_ref, atol=1.0e-6))

  def test_conv_forward(self):
    r''' Tests octree2col/col2octree, octree_conv/octree_deconv.
    '''

    folder = os.path.dirname(__file__)
    data = np.load(os.path.join(folder, 'data/octree_nn.npz'))
    octree = get_batch_octree()

    depth = data['depth'].item()
    in_channels = data['channel'].item()
    out_channels = data['channel_out'].item()
    stride = data['stride']
    kernel_size = data['kernel_size']
    data_in = [torch.from_numpy(data['data_0']),
               torch.from_numpy(data['data_1'])]

    counter = 0
    for i in range(len(stride)):
      for j in range(len(kernel_size)):
        for ne in [True, False]:
          # test octree2col
          kernel = '{}{}{}'.format(
              kernel_size[j][0], kernel_size[j][1], kernel_size[j][2])
          o2c = ocnn.nn.octree2col(
              data_in[ne], octree, depth, kernel, stride[i], ne)
          gt = data['o2c_%d' % counter]
          self.assertTrue(
              np.array_equal(o2c.numpy(), gt), 'counter: %d' % counter)

          # test col2octree
          c2o = ocnn.nn.col2octree(o2c, octree, depth, kernel, stride[i], ne)
          gt = data['c2o_%d' % counter]
          self.assertTrue(
              np.array_equal(c2o.numpy(), gt), 'counter: %d' % counter)

          # update counter
          counter = counter + 1

    for m in ['explicit_gemm', 'block_gemm']:
      counter = 0
      for i in range(len(stride)):
        for j in range(len(kernel_size)):
          for ne in [True, False]:
            # test octree_conv
            conv = ocnn.nn.OctreeConv(
                in_channels, out_channels, kernel_size[j].tolist(), stride[i],
                nempty=ne, method=m, max_buffer=int(2e4))
            weight = torch.from_numpy(data['cw_%d' % counter])
            conv.weights.data.copy_(weight)
            out = conv.forward(data_in[ne], octree, depth)
            gt = data['conv_%d' % counter]
            self.assertTrue(np.allclose(out.detach().numpy(), gt, atol=1e-6))

            # test octree_deconv
            deconv = ocnn.nn.OctreeDeconv(
                in_channels, out_channels, kernel_size[j].tolist(), stride[i],
                nempty=ne, method=m, max_buffer=int(2e4))
            weight = torch.from_numpy(data['dw_%d' % counter])
            deconv.weights.data.copy_(weight)
            out = deconv.forward(data_in[ne], octree, depth)
            gt = data['dconv_%d' % counter]
            self.assertTrue(np.allclose(out.detach().numpy(), gt, atol=1e-6))

            # update counter
            counter = counter + 1

  def test_conv_backward(self):

    folder = os.path.dirname(__file__)
    data = np.load(os.path.join(folder, 'data/octree_nn.npz'))
    octree = get_batch_octree()

    depth = data['depth'].item()
    in_channels = data['channel'].item()
    out_channels = data['channel_out'].item()
    stride = data['stride']
    kernel_size = data['kernel_size']
    data_in = [torch.from_numpy(data['data_0']),
               torch.from_numpy(data['data_1'])]

    counter = 0
    for i in range(len(stride)):
      for j in range(len(kernel_size)):
        for ne in [True, False]:

          # test octree_conv
          conv_ref = ocnn.nn.OctreeConv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i],
              nempty=ne, method='explicit_gemm')
          weight = torch.from_numpy(data['cw_%d' % counter])
          conv_ref.weights.data.copy_(weight)
          data_ref = data_in[ne].clone().requires_grad_()
          out_ref = conv_ref.forward(data_ref, octree, depth)
          loss_ref = out_ref.sum()
          loss_ref.backward()

          conv = ocnn.nn.OctreeConv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i],
              nempty=ne, method='block_gemm', max_buffer=int(2e4))
          weight = torch.from_numpy(data['cw_%d' % counter])
          conv.weights.data.copy_(weight)
          data_ = data_in[ne].clone().requires_grad_()
          out = conv.forward(data_, octree, depth)
          loss = out.sum()
          loss.backward()

          self.assertTrue(np.allclose(
              out.data.numpy(), out_ref.data.numpy(), atol=1e-6))
          self.assertTrue(np.allclose(
              data_.grad.numpy(), data_ref.grad.numpy(), atol=1e-6))
          self.assertTrue(np.allclose(
              conv.weights.grad.numpy(), conv_ref.weights.grad.numpy(),
              atol=1e-6))

          # test octree_deconv
          deconv_ref = ocnn.nn.OctreeDeconv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i],
              nempty=ne, method='explicit_gemm')
          weight = torch.from_numpy(data['dw_%d' % counter])
          deconv_ref.weights.data.copy_(weight)
          data_ref = data_in[ne].clone().requires_grad_()
          out_ref = deconv_ref.forward(data_ref, octree, depth)
          loss_ref = out_ref.sum()
          loss_ref.backward()

          deconv = ocnn.nn.OctreeDeconv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i],
              nempty=ne, method='block_gemm', max_buffer=int(2e4))
          weight = torch.from_numpy(data['dw_%d' % counter])
          deconv.weights.data.copy_(weight)
          data_ = data_in[ne].clone().requires_grad_()
          out = deconv.forward(data_, octree, depth)
          loss = out.sum()
          loss.backward()

          self.assertTrue(np.allclose(
              out.data.numpy(), out_ref.data.numpy(), atol=1e-6))
          self.assertTrue(np.allclose(
              data_.grad.numpy(), data_ref.grad.numpy(), atol=1e-6))
          self.assertTrue(np.allclose(
              deconv.weights.grad.numpy(), deconv_ref.weights.grad.numpy(),
              atol=1e-6))

          # update counter
          counter = counter + 1


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
