import os
import torch
import unittest

import ocnn


class ShuffledKeyTest(unittest.TestCase):

  def test_shuffled_key(self):
    devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
    for d in devices:
      x = torch.randint(65536, (10000,), device=d)
      y = torch.randint(65536, (10000,), device=d)
      z = torch.randint(65536, (10000,), device=d)
      b = torch.randint(32768, (10000,), device=d)

      key = ocnn.octree.xyz2key(x, y, z, b)
      x1, y1, z1, b1 = ocnn.octree.key2xyz(key)

      self.assertTrue((x1 == x).all() & (y1 == y).all() &
                      (z1 == z).all() & (b1 == b).all())


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
