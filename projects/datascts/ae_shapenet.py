# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from ocnn.octree import Points
from ocnn.dataset import CollateBatch
from thsolver import Dataset

from .utils import Transform
from .image2shape import ReadPoints


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


def get_ae_shapenet_dataset(flags):
  transform = ShapeNetTransform(flags)
  read_file = ReadPoints(has_normal=True, filetype=flags.filetype)
  collate_batch = CollateBatch(merge_points=False)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_batch
