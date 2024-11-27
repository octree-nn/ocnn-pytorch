# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import numpy as np
from thsolver import Dataset
from ocnn.octree import Points
from ocnn.dataset import CollateBatch

from .utils import Transform


label_name_mapping = {
    0: 'unlabeled', 1: 'outlier', 10: 'car', 11: 'bicycle', 13: 'bus',
    15: 'motorcycle', 16: 'on-rails', 18: 'truck', 20: 'other-vehicle',
    30: 'person', 31: 'bicyclist', 32: 'motorcyclist', 40: 'road',
    44: 'parking', 48: 'sidewalk', 49: 'other-ground', 50: 'building',
    51: 'fence', 52: 'other-structure', 60: 'lane-marking', 70: 'vegetation',
    71: 'trunk', 72: 'terrain', 80: 'pole', 81: 'traffic-sign',
    99: 'other-object', 252: 'moving-car', 253: 'moving-bicyclist',
    254: 'moving-person', 255: 'moving-motorcyclist', 256: 'moving-on-rails',
    257: 'moving-bus', 258: 'moving-truck', 259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


def get_label_map():
  num_classes = len(kept_labels)  # = 19
  label_ids = list(range(1, num_classes + 1))
  label_dict = dict(zip(kept_labels, label_ids))

  label_map = np.zeros(260)
  for idx, name in label_name_mapping.items():
    name = name.replace('moving-', '')
    label_map[idx] = label_dict.get(name, 0)
  return label_map


class KittiTransform(Transform):

  def __init__(self, flags):
    super().__init__(flags)

    self.scale_factor = 100
    self.label_map = get_label_map()

  def preprocess(self, sample, idx=None):
    # get the input
    xyz = sample['points'][:, :3]
    density = sample['points'][:, 3:]

    # normalization the xyz to [-1, 1]
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
    xyz = (xyz - center) / self.scale_factor

    # remap the labels
    labels = sample['labels']
    labels = self.label_map[labels & 0xFFFF].astype(np.float32)

    points = Points(
        torch.from_numpy(xyz), None, torch.from_numpy(density),
        torch.from_numpy(labels).unsqueeze(1))
    return {'points': points}


def read_file(filename):
  points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)

  label_name = filename.replace('velodyne', 'labels').replace('.bin', '.label')
  if os.path.exists(label_name):
    labels = np.fromfile(label_name, dtype=np.int32).reshape(-1)
  else:
    labels = np.zeros((points.shape[0],), dtype=np.int32)

  return {'points': points, 'labels': labels}


def get_kitti_dataset(flags):
  transform = KittiTransform(flags)
  collate_batch = CollateBatch(merge_points=True)
  dataset = Dataset(flags.location, flags.filelist, transform, read_file)
  return dataset, collate_batch
