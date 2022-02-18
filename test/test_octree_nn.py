import os
import torch
import numpy as np
import unittest

import ocnn
from .utils import get_batch_octree


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
        for e in [True, False]:
          # test octree2col
          kernel = '{}{}{}'.format(
              kernel_size[j][0], kernel_size[j][1], kernel_size[j][2])
          o2c = ocnn.nn.octree2col(
              data_in[e], octree, depth, kernel, stride[i], e)
          gt = data['o2c_%d' % counter]
          self.assertTrue(
              np.array_equal(o2c.numpy(), gt), 'counter: %d' % counter)

          # test col2octree
          c2o = ocnn.nn.col2octree(o2c, octree, depth, kernel, stride[i], e)
          gt = data['c2o_%d' % counter]
          self.assertTrue(
              np.array_equal(c2o.numpy(), gt), 'counter: %d' % counter)

          # test octree_conv
          conv = ocnn.nn.OctreeConv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i], e)
          weight = torch.from_numpy(data['cw_%d' % counter])
          conv.weights.data.copy_(weight)
          out = conv.forward(data_in[e], octree, depth)
          gt = data['conv_%d' % counter]
          self.assertTrue(
              np.allclose(out.detach().numpy(), gt, atol=1e-6), 'counter: %d' % counter)

          # test octree_deconv
          deconv = ocnn.nn.OctreeDeconv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i], e)
          weight = torch.from_numpy(data['dw_%d' % counter])
          deconv.weights.data.copy_(weight)
          out = deconv.forward(data_in[e], octree, depth)
          gt = data['dconv_%d' % counter]
          self.assertTrue(
              np.allclose(out.detach().numpy(), gt, atol=1e-6), 'counter: %d' % counter)

          counter = counter + 1

    # max pool
    pool, idx = ocnn.nn.octree_max_pool(
        data_in[0], octree, depth, return_indices=True)
    self.assertTrue(np.array_equal(pool.numpy(), data['pool']))
    upool = ocnn.nn.octree_max_unpool(pool, idx, octree, depth-1)
    self.assertTrue(np.array_equal(upool.numpy(), data['upool']))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
