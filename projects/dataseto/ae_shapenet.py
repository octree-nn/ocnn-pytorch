# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import numpy as np
from ocnn.octree import Points
from ocnn.dataset import CollateBatch
from thsolver import Dataset

from .utils import ReadPly, Transform


class ShapeNetTransform(Transform):

  def __init__(self, flags):
    super().__init__(flags)
    self.points_scale = flags.points_scale

  def preprocess(self, sample, idx):
    points = torch.from_numpy(sample['points'])
    normals = torch.from_numpy(sample['normals'])
    points = points / self.points_scale   # scale to [-1.0, 1.0]

    point_cloud = Points(points, normals)
    return {'points': point_cloud}


class ReadFile:
  def __init__(self, has_normal: bool = True):
    self.read_ply = ReadPly(has_normal, has_color=False, has_label=False)

  def __call__(self, filename):
    return self.read_ply(filename + '.ply')


def get_ae_shapenet_dataset(flags):
  transform = ShapeNetTransform(flags)
  read_file = ReadFile(has_normal=True)
  collate_batch = CollateBatch(merge_points=False)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_batch
