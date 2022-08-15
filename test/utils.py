# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import numpy as np

import ocnn


def get_points(id, return_data=False):
  folder = os.path.dirname(__file__)
  filename = os.path.join(folder, 'data/octree/test_%03d.npz' % id)
  data = np.load(filename)

  points, normals = data['points'], data['normals']
  point_cloud = ocnn.octree.Points(
      torch.from_numpy(points), torch.from_numpy(normals))
  return (point_cloud, data) if return_data else point_cloud


def get_octree(id, return_data=False):
  point_cloud, data = get_points(id, return_data=True)
  octree = ocnn.octree.Octree(
      data['depth'].item(), full_depth=data['full_depth'].item())
  octree.build_octree(point_cloud)
  return (octree, data) if return_data else octree


def get_batch_octree(device='cpu'):
  octree1 = get_octree(4).to(device)
  octree2 = get_octree(5).to(device)
  octree = ocnn.octree.merge_octrees([octree1, octree2])
  octree.construct_all_neigh()
  return octree
