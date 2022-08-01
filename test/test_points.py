# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import torch.nn.functional as F
import unittest
import math

import ocnn


class TestPoints(unittest.TestCase):

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

  def test_uniform_scale(self):
    point_cloud = self.init_points()

    factor = torch.Tensor([0.5, 0.5, 0.5])
    normals = point_cloud.normals.clone()
    points = point_cloud.points.clone()
    point_cloud.scale(factor)
    self.assertTrue((point_cloud.normals == normals).all() &
                    (point_cloud.points == points * 0.5).all())

  def test_scale(self):
    point_cloud = self.init_points()

    factor = torch.Tensor([1.0, 2.0, 3.0])
    normals = point_cloud.normals.clone()
    points = point_cloud.points.clone()
    point_cloud.scale(factor)
    self.assertTrue(torch.allclose(point_cloud.normals, F.normalize(normals/factor)) &
                    torch.equal(point_cloud.points, points * factor))

  def test_rotation(self):
    point_cloud = self.init_points()

    # rot x
    angle = torch.Tensor([math.pi / 2.0, 0.0, 0.0])
    normals = point_cloud.normals.clone()
    points = point_cloud.points.clone()
    points = torch.stack([points[:, 0], -points[:, 2], points[:, 1]], dim=1)
    normals = torch.stack([normals[:, 0], -normals[:, 2], normals[:, 1]], dim=1)

    point_cloud.rotate(angle)
    self.assertTrue(torch.allclose(point_cloud.normals, normals, atol=1e-6) &
                    torch.allclose(point_cloud.points, points, atol=1e-6))

    # rot y
    angle = torch.Tensor([0.0, math.pi / 2.0, 0.0])
    points = torch.stack([points[:, 2], points[:, 1], -points[:, 0]], dim=1)
    normals = torch.stack([normals[:, 2], normals[:, 1], -normals[:, 0]], dim=1)

    point_cloud.rotate(angle)
    self.assertTrue(torch.allclose(point_cloud.normals, normals, atol=1e-6) &
                    torch.allclose(point_cloud.points, points, atol=1e-6))

    # rot z
    angle = torch.Tensor([0.0, 0.0, math.pi / 2.0])
    points = torch.stack([-points[:, 1], points[:, 0], points[:, 2]], dim=1)
    normals = torch.stack([-normals[:, 1], normals[:, 0], normals[:, 2]], dim=1)

    point_cloud.rotate(angle)
    self.assertTrue(torch.allclose(point_cloud.normals, normals, atol=1e-6) &
                    torch.allclose(point_cloud.points, points, atol=1e-6))

  def test_normalize(self):
    point_cloud = self.init_points()

    bbmin, bbmax = point_cloud.bbox()
    point_cloud.normalize(bbmin, bbmax)

    self.assertTrue((point_cloud.points >= -1).all() &
                    (point_cloud.points <= 1).all())


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
