import ocnn
import torch
import numpy as np

from ocnn.octree import Octree, Points
from ocnn.dataset import CollateBatch
from thsolver import Dataset
from .utils import ReadPly


class Transform:

  def __init__(self, flags):
    super().__init__()
    self.depth = flags.depth
    self.full_depth = flags.full_depth

    self.points_number = 3000
    self.points_scale = 0.95
    self.noise_std = 0.01 * self.points_scale

  def points2octree(self, points: Points):
    octree = Octree(self.depth, self.full_depth)
    octree.build_octree(points)
    return octree

  def __call__(self, point_cloud, idx):
    # get the input
    points, normals = point_cloud['points'], point_cloud['normals']

    # normalize the points
    bbmin, bbmax = np.min(points, axis=0), np.max(points, axis=0)
    center = (bbmin + bbmax) / 2.0
    radius = 2.0 / (np.max(bbmax - bbmin) + 1.0e-6)
    points = (points - center) * radius  # normalize to [-1, 1]
    points *= self.points_scale  # normalize to [-points_scale, points_scale]

    # randomly sample points and add noise
    noise = self.noise_std * np.random.randn(self.points_number, 3)
    rand_idx = np.random.choice(points.shape[0], size=self.points_number)
    points_noise = points[rand_idx] + noise

    # transform points to octree
    points_gt = Points(
        torch.from_numpy(points).float(), torch.from_numpy(normals).float())
    points_gt.clip(min=-1, max=1)
    octree_gt = self.points2octree(points_gt)

    points_in = Points(torch.from_numpy(points_noise).float())
    points_in.clip(min=-1, max=1)
    octree_in = self.points2octree(points_in)

    return {'octree': octree_in, 'points': points_in,
            'octree_gt': octree_gt, 'points_gt': points_gt}


class ReadFile:
  def __init__(self, has_normal: bool = True):
    self.has_normal = has_normal
    self.read_ply = ReadPly(has_normal, has_color=False, has_label=False)

  def __call__(self, filename):
    if filename.endswith('.npz'):
      raw = np.load(filename)
    elif filename.endswith('.ply'):
      raw = self.read_ply(filename)
    else:
      raise ValueError
    return raw


def get_completion_dataset(flags):
  transform = Transform(flags)
  read_file = ReadFile(has_normal=True)
  collate_batch = CollateBatch(merge_points=False)
  dataset = Dataset(flags.location, flags.filelist, transform, read_file)
  return dataset, collate_batch
