# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch

from solver import Dataset
from ocnn.octree import Points
from ocnn.dataset import CollateBatch

from .utils import ReadPly, Transform


class ShapeNetTransform(Transform):

  def preprocess(self, sample: dict, idx: int):

    xyz = torch.from_numpy(sample['points']).float()
    normal = torch.from_numpy(sample['normals']).float()
    labels = torch.from_numpy(sample['labels']).float()
    points = Points(xyz, normal, labels=labels)

    # !NOTE: Normalize the points into one unit sphere in [-0.8, 0.8]
    bbmin, bbmax = points.bbox()
    points.normalize(bbmin, bbmax, scale=0.8)

    return points


def get_seg_shapenet_dataset(flags):
  transform = ShapeNetTransform(flags)
  read_ply = ReadPly(has_normal=True, has_label=True)
  collate_batch = CollateBatch(merge_points=True)

  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_ply, take=flags.take)
  return dataset, collate_batch
