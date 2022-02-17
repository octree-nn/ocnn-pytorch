import os
import torch
import numpy as np
import unittest

import ocnn
from .utils import get_octree, get_batch_octree


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
          octree2col = ocnn.nn.Octree2Col(kernel, stride[i], e)
          out = octree2col.forward(data_in[e], octree, depth)

          gt = data['out_%d' % counter]
          self.assertTrue(
              np.array_equal(out.numpy(), gt), 'counter: %d' % counter)

          counter = counter + 1


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
