import os
import torch
import numpy as np
import unittest

import ocnn
from .utils import get_batch_octree


class TesOctree2Col(unittest.TestCase):

  def test_octree2col(self):

    folder = os.path.dirname(__file__)
    data = np.load(os.path.join(folder, 'data/octree2col.npz'))
    octree = get_batch_octree()

    depth = data['depth'].item()
    stride = data['stride']
    kernel_size = data['kernel_size']
    data_in = [torch.from_numpy(data['data_0']),
               torch.from_numpy(data['data_1'])]

    counter = 0
    for i in range(len(stride)):
      for j in range(len(kernel_size)):
        for e in [True, False]:
          kernel = '{}{}{}'.format(
              kernel_size[j][0], kernel_size[j][1], kernel_size[j][2])
          o2c = ocnn.nn.octree2col(
              data_in[e], octree, depth, kernel, stride[i], e)
          gt = data['o2c_%d' % counter]
          self.assertTrue(
              np.array_equal(o2c.numpy(), gt), 'counter: %d' % counter)

          c2o = ocnn.nn.col2octree(o2c, octree, depth, kernel, stride[i], e)
          gt = data['c2o_%d' % counter]
          self.assertTrue(
              np.array_equal(c2o.numpy(), gt), 'counter: %d' % counter)

          counter = counter + 1


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
