import os
import torch
import unittest

import ocnn


class TestScatter(unittest.TestCase):

  def init_points(self):
    s2, s3 = 2.0 ** 0.5 / 2.0, 3.0 ** 0.5 / 3.0
    points = torch.Tensor([[1, 2, 3], [-4, -5, -6]])
    normals = torch.Tensor([[s2, -s2, 0], [-s3, -s3, s3]])
    labels = torch.Tensor([[1], [2]])
    features = torch.rand(2, 4)
    return ocnn.octree.Points(points, normals, features, labels)

  def test_orient_normal(self):
    point_cloud = self.init_points()
    normals = point_cloud.normals.clone()
    normals[1] = -normals[1]  # orient the 2nd normal
    point_cloud.orient_normal('x')
    self.assertTrue((point_cloud.normals == normals).all())

  def test_scale(self):
    point_cloud = self.init_points()

    factor = torch.Tensor([0.5, 0.5, 0.5])    
    normals = point_cloud.normals.clone()
    points = point_cloud.points.clone()
    point_cloud.scale(factor)
    self.assertTrue((point_cloud.normals == normals).all() &
                    (point_cloud.points == points * 0.5).all())

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
