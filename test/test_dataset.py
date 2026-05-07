# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import unittest

import numpy as np

import ocnn


class TestTransform(unittest.TestCase):

  def test_transform_without_normals(self):
    transform = ocnn.dataset.Transform(
        depth=2, full_depth=1, distort=False, angle=[0, 0, 0],
        interval=[1, 1, 1], scale=0.0, uniform=True, jitter=0.0,
        flip=[0.0, 0.0, 0.0])
    sample = {
        'points': np.array(
            [[0.0, 0.0, 0.0], [0.5, -0.5, 0.25]], dtype=np.float32),
        'label': 3,
    }

    output = transform(sample, idx=0)

    self.assertIsNone(output['points'].normals)
    self.assertIsNone(output['octree'].normals[transform.depth])
    self.assertEqual(output['label'], 3)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
