import os
import torch
import torch.nn.functional as F
import unittest
import math

import ocnn


class TesOctree(unittest.TestCase):

  def init_points(self):
    points = torch.Tensor([[-1, -1, -1], [0, 0, -1], [0.0625, 0.0625, -1]])
    normals = torch.Tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
    features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
    labels = torch.Tensor([[0], [2], [2]])
    return ocnn.octree.Points(points, normals, features, labels)

  def test_build_octree(self):
    point_cloud = self.init_points()
    octree = ocnn.octree.Octree(depth=5, full_depth=1)
    octree.build_octree(point_cloud)

    # test node number
    nnum = torch.Tensor([1, 8, 16, 16, 16, 16])
    nnum_nempty = torch.Tensor([1, 2, 2, 2, 2, 3])
    self.assertTrue((octree.nnum == nnum).all())
    self.assertTrue((octree.nnum_nempty == nnum_nempty).all())

    # test the key
    keys = [
        torch.Tensor([0]),
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 48,
                     49, 50, 51, 52, 53, 54, 55, ]),
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 384, 385,
                     386, 387, 388, 389, 390, 391, ]),
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 3072, 3073,
                     3074, 3075, 3076, 3077, 3078, 3079, ]),
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 24576, 24577,
                      24578, 24579, 24580, 24581, 24582, 24583]), ]
    for d in range(0, 6):
      self.assertTrue((octree.keys[d] == keys[d]).all())

    # test the children
    children = [
        torch.Tensor([0]),
        torch.Tensor([0, -1, -1, -1, -1, -1, 1, -1]),
        torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1,
                      1, -1, -1, -1, -1, -1, -1, -1]),
        torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1,
                      1, -1, -1, -1, -1, -1, -1, -1]),
        torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1,
                      1, -1, -1, -1, -1, -1, -1, -1]),
        torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1,
                      1, -1, -1, -1, -1, -1, 2, -1]), ]

    for d in range(0, 6):
      self.assertTrue((octree.children[d] == children[d]).all())

    #  test the signal
    normals = torch.Tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
    features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
    self.assertTrue((octree.normals[5] == normals).all())
    self.assertTrue((octree.features[5] == features).all())


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
