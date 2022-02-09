import os
import torch
import unittest

import ocnn


class ShuffledKeyTest(unittest.TestCase):

  def test_shuffled_key(self):
    x = torch.tensor([2049, 511])
    y = torch.tensor([4095, 4097])
    z = torch.tensor([8011, 8009])
    b = torch.tensor([0, 1])
    key = ocnn.octree.xyz2key(x, y, z, b)
    x1, y1, z1, b1 = ocnn.octree.key2xyz(key)

    self.assertTrue((x1 == x).all() & (y1 == y).all() &
                    (z1 == z).all() & (b1 == b).all())


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
