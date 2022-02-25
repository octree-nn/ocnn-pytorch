import torch
import numpy as np
from plyfile import PlyData
from typing import List

from solver import Dataset
from ocnn.octree import Points, Octree, merge_octrees


class TransformModelNet:

  def __init__(self, flags):
    self.flags = flags

  def __call__(self, points: Points, idx: int):
    # Normalize the points into one unit sphere in [-0.75, 0.75]
    bbmin, bbmax = points.bbox()
    points.normalize(bbmin, bbmax, scale=0.75)

    # Apply the general transformations provided by ocnn.
    # The augmentations including rotation, scaling, and jittering.
    if self.flags.distort:
      rng_angle, rng_scale, rng_jitter = self.rnd_parameters()
      points.rotate(rng_angle)
      points.scale(rng_scale)
      points.translate(rng_jitter)

    # Orient normals since the original meshes contains flipped triangles
    points.orient_normal('xyz')
    # !!! Clip to [-1, 1] before octree building
    points.clip(min=-1, max=1)

    # Convert the points to an octree
    octree = Octree(self.flags.depth, self.flags.full_depth)
    octree.build_octree(points)

    return {'octree': octree, 'points': points}

  def rnd_parameters(self):
    flags = self.flags

    rnd_angle = [None] * 3
    for i in range(3):
      rot_num = flags.angle[i] // flags.interval[i]
      rnd = torch.randint(low=-rot_num, high=rot_num+1, size=(1,))
      rnd_angle[i] = rnd * flags.interval[i] * (3.14159265 / 180.0)
    rnd_angle = torch.cat(rnd_angle)

    rnd_scale = torch.rand(3) * (2 * flags.scale) - flags.scale + 1.0
    if flags.uniform:
      rnd_scale[1] = rnd_scale[0]
      rnd_scale[2] = rnd_scale[0]

    rnd_jitter = torch.rand(3) * (2 * flags.jitter) - flags.jitter

    return rnd_angle, rnd_scale, rnd_jitter


def read_ply(filename: str):
  plydata = PlyData.read(filename)
  vtx = plydata['vertex']
  points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)
  normal = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
  point_cloud = Points(torch.from_numpy(points.astype(np.float32)),
                       torch.from_numpy(normal.astype(np.float32)))
  return point_cloud


def collate_octrees(batch: List):
  assert type(batch) == list

  outputs = {}
  for key in batch[0].keys():
    outputs[key] = [b[key] for b in batch]

    # Merge a batch of octrees into one super octree
    if 'octree' in key:
      octree = merge_octrees(outputs[key])
      octree.construct_all_neigh()
      outputs[key] = octree

    # Convert the labels to a Tensor
    if 'label' in key:
      outputs['label'] = torch.tensor(outputs[key])

  return outputs


def get_modelnet40_dataset(flags):
  transform = TransformModelNet(flags)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_ply,
                    in_memory=flags.in_memory)
  return dataset, collate_octrees
