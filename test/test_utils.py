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
      self.assertTrue(torch.equal(output, gt))

  def test_cumsum(self):
    data = torch.tensor([[1, 2, 3], [4, 5, 6]])
    gt1 = torch.tensor([[1, 3, 6], [4, 9, 15]])
    gt2 = torch.tensor([[0, 1, 3, 6], [0, 4, 9, 15]])
    gt3 = torch.tensor([[0, 0, 0], [1, 2, 3], [5, 7, 9]])

    out1 = ocnn.utils.cumsum(data, dim=1, exclusive=False)
    out2 = ocnn.utils.cumsum(data, dim=1, exclusive=True)
    out3 = ocnn.utils.cumsum(data, dim=0, exclusive=True)
    self.assertTrue(torch.equal(gt1, out1))
    self.assertTrue(torch.equal(gt2, out2))
    self.assertTrue(torch.equal(gt3, out3))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
