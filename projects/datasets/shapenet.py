import os
import torch
import numpy as np

from ocnn.octree import Octree, Points
from ocnn.dataset import CollateBatch
from thsolver import Dataset


class Transform:

  def __init__(self, flags):
    super().__init__()
    self.depth = flags.depth
    self.full_depth = flags.full_depth
    self.points_scale = flags.points_scale

  def points2octree(self, points: Points):
    octree = Octree(self.depth, self.full_depth)
    octree.build_octree(points)
    return octree

  def __call__(self, point_cloud, idx):
    # get the input
    points = point_cloud['points']
    normals = point_cloud.get('normals', np.array([]))

    # normalize the points
    bbmin, bbmax = np.min(points, axis=0), np.max(points, axis=0)
    center = (bbmin + bbmax) / 2.0
    radius = 2.0 / (np.max(bbmax - bbmin) + 1.0e-6)
    points = (points - center) * radius  # normalize to [-1, 1]
    points *= self.points_scale          # normalize to [-points_scale, points_scale]

    # transform points to octree
    points_in = Points(torch.from_numpy(points).float(),
                       torch.from_numpy(normals).float())
    points_in.clip(min=-1, max=1)
    octree_in = self.points2octree(points_in)
    return {'points': points_in, 'octree': octree_in, }


def read_file(filename):
  filename = os.path.join(filename, 'pointcloud.npz')
  assert filename.endswith('.npz') or filename.endswith('.npy')
  return np.load(filename)


def get_shapenet_dataset(flags):
  transform = Transform(flags)
  collate_batch = CollateBatch(merge_points=False)
  dataset = Dataset(flags.location, flags.filelist, transform, read_file)
  return dataset, collate_batch
