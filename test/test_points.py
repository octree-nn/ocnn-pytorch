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
    self.assertTrue(torch.allclose(point_cloud.normals, F.normalize(
        normals/factor)) & torch.equal(point_cloud.points, points * factor))

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

  def test_clip(self):
    point_cloud = self.init_points()
    point_cloud.clip(min=-4, max=4)

    s2 = 2.0 ** 0.5 / 2.0
    self.assertTrue(torch.equal(
        point_cloud.points, torch.Tensor([[1, 2, 3]])))
    self.assertTrue(torch.equal(
        point_cloud.normals, torch.Tensor([[s2, -s2, 0]])))
    self.assertTrue(torch.equal(
        point_cloud.labels, torch.Tensor([[1]])))

  def test_split_points_unequal_batches(self):
    points = torch.randn(7, 3)
    features = torch.randn(7, 8)
    normals = torch.randn(7, 3)
    labels = torch.randint(0, 5, (7, 1))
    batch_id = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    bnd = [0, 2, 5, 7]
    point_cloud = ocnn.octree.Points(
        points, normals, features, labels, batch_id=batch_id, batch_size=3)

    outs = point_cloud.split_points()

    # Verify split sizes: [2, 3, 2]
    self.assertEqual(len(outs), 3)
    self.assertEqual(outs[0].npt, 2)
    self.assertEqual(outs[1].npt, 3)
    self.assertEqual(outs[2].npt, 2)

    # Verify correct points are in each split
    for i in range(3):
      rng = range(bnd[i], bnd[i+1])
      self.assertTrue(torch.equal(outs[i].points, points[rng]))
      self.assertTrue(torch.equal(outs[i].normals, normals[rng]))
      self.assertTrue(torch.equal(outs[i].features, features[rng]))
      self.assertTrue(torch.equal(outs[i].labels, labels[rng]))

  def test_split_points_single_batch(self):
    # Test with a single batch (should return list with one element)
    points = torch.randn(5, 3)
    batch_id = torch.zeros(5, 1, dtype=torch.long)

    point_cloud = ocnn.octree.Points(
        points, batch_id=batch_id, batch_size=1)

    out = point_cloud.split_points()
    self.assertEqual(len(out), 1)
    self.assertEqual(out[0].npt, 5)
    self.assertTrue(torch.equal(out[0].points, points))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
