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
from PIL import Image
from torchvision.transforms import ToTensor

from .utils import ReadPly, Transform


class Image2ShapeTransform(Transform):

  def __init__(self, flags):
    super().__init__(flags)
    self.points_scale = flags.points_scale
    self.img2tensor = ToTensor()

  def preprocess(self, sample, idx):
    points = torch.from_numpy(sample['points'])
    normals = torch.from_numpy(sample['normals'])
    points = points / self.points_scale   # scale to [-1.0, 1.0]
    point_cloud = Points(points, normals)

    # convert image to tensor
    image = self.img2tensor(sample['image'])
    return {'points': point_cloud, 'image': image}


class ReadFile:
  def __init__(self, has_normal: bool = True):
    self.has_normal = has_normal
    self.read_ply = ReadPly(has_normal, has_color=False, has_label=False)

  def __call__(self, filename):
    # read points from a ply file
    filename_ply = filename + '.ply'
    output = self.read_ply(filename_ply)

    # read images from a png file
    rnd_idx = np.random.randint(0, 24)
    folder_img = filename_ply.replace('points.ply', 'renderings')[:-4]
    filename_img = folder_img + ('/rendering/%02d.png' % rnd_idx)
    output['image'] = Image.open(filename_img).convert('RGB')  # rgba -> rgb
    return output


class Image2ShapeCollateBatch(CollateBatch):
  def __init__(self, merge_points: bool = False):
    super().__init__(merge_points)

  def __call__(self, batch):
    output = super().__call__(batch)

    # merge images to a tensor
    output['image'] = torch.stack(output['image'], dim=0)
    return output


def get_image2shape_dataset(flags):
  transform = Image2ShapeTransform(flags)
  read_file = ReadFile(has_normal=True)
  collate_batch = Image2ShapeCollateBatch(merge_points=False)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_batch
