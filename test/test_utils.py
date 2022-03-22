import os
import torch
import unittest

import ocnn


class TestScatter(unittest.TestCase):

  def test_scatter_add(self):
    devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
    for device in devices:
      src = torch.arange(1, 11, device=device).view(2, 5)
      idx = torch.tensor([0, 1, 3, 2, 0], device=device)
      gt = torch.tensor([[6, 2, 4, 3, 0], [16, 7, 9, 8, 0]], device=device)

      output = ocnn.utils.scatter_add(src, idx, dim=1, dim_size=5)
      self.assertTrue((output == gt).all())


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
