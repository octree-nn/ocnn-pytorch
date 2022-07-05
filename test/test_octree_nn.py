import os
import torch
import numpy as np
import unittest

import ocnn
from .utils import get_batch_octree, get_octree


class TesOctreeNN(unittest.TestCase):

  def test_octree_nn(self):

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

    for m in [True, False]:
      counter = 0
      for i in range(len(stride)):
        for j in range(len(kernel_size)):
          for ne in [True, False]:
            # test octree_conv
            conv = ocnn.nn.OctreeConv(
                in_channels, out_channels, kernel_size[j].tolist(), stride[i],
                nempty=ne, direct_method=m, max_buffer=int(2e4))
            weight = torch.from_numpy(data['cw_%d' % counter])
            conv.weights.data.copy_(weight)
            out = conv.forward(data_in[ne], octree, depth)
            gt = data['conv_%d' % counter]
            self.assertTrue(np.allclose(out.detach().numpy(), gt, atol=1e-6))

            # test octree_deconv
            deconv = ocnn.nn.OctreeDeconv(
                in_channels, out_channels, kernel_size[j].tolist(), stride[i],
                nempty=ne, direct_method=m, max_buffer=int(2e4))
            weight = torch.from_numpy(data['dw_%d' % counter])
            deconv.weights.data.copy_(weight)
            out = deconv.forward(data_in[ne], octree, depth)
            gt = data['dconv_%d' % counter]
            self.assertTrue(np.allclose(out.detach().numpy(), gt, atol=1e-6))

            # update counter
            counter = counter + 1

    # test max pool
    pool, idx = ocnn.nn.octree_max_pool(
        data_in[0], octree, depth, return_indices=True)
    self.assertTrue(np.array_equal(pool.numpy(), data['pool']))
    upool = ocnn.nn.octree_max_unpool(pool, idx, octree, depth-1)
    self.assertTrue(np.array_equal(upool.numpy(), data['upool']))

  def test_conv_backward_nn(self):

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
              nempty=ne, direct_method=True)
          weight = torch.from_numpy(data['cw_%d' % counter])
          conv_ref.weights.data.copy_(weight)
          data_ref = data_in[ne].clone().requires_grad_()
          out_ref = conv_ref.forward(data_ref, octree, depth)
          loss_ref = out_ref.sum()
          loss_ref.backward()

          conv = ocnn.nn.OctreeConv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i],
              nempty=ne, direct_method=False, max_buffer=int(2e4))
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
              nempty=ne, direct_method=True)
          weight = torch.from_numpy(data['dw_%d' % counter])
          deconv_ref.weights.data.copy_(weight)
          data_ref = data_in[ne].clone().requires_grad_()
          out_ref = deconv_ref.forward(data_ref, octree, depth)
          loss_ref = out_ref.sum()
          loss_ref.backward()

          deconv = ocnn.nn.OctreeDeconv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i],
              nempty=ne, direct_method=False, max_buffer=int(2e4))
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
