import torch
import numpy as np
from plyfile import PlyData
from typing import List

from solver import Dataset
from ocnn.octree import Points, Octree, merge_octrees, merge_points


class Transform:

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

    if self.flags.orient_normal:
      # Orient normals since the original meshes contains flipped triangles
      points.orient_normal(self.flags.orient_normal)
    # !!! NOTE: Clip to [-1, 1] before octree building
    inbox_mask = points.clip(min=-1, max=1)

    # Convert the points to an octree
    octree = Octree(self.flags.depth, self.flags.full_depth)
    octree.build_octree(points)

    return {'octree': octree, 'points': points, 'inbox_mask': inbox_mask}

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


class ReadPly:

  def __init__(self, has_normal: bool = True, has_color: bool = False,
               has_label: bool = False):
    self.has_normal = has_normal
    self.has_color = has_color
    self.has_label = has_label

  def __call__(self, filename: str):
    plydata = PlyData.read(filename)
    vtx = plydata['vertex']

    kwargs = dict()
    points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)
    kwargs['points'] = torch.from_numpy(points.astype(np.float32))
    if self.has_normal:
      normal = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
      kwargs['normals'] = torch.from_numpy(normal.astype(np.float32))
    if self.has_color:
      color = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=1)
      kwargs['colors'] = torch.from_numpy(color.astype(np.float32))
    if self.has_label:
      label = vtx['label']
      kwargs['labels'] = torch.from_numpy(label.astype(np.int32))

    return Points(**kwargs)


class CollateBatch:

  def __init__(self, merge_points: bool = False):
    self.merge_points = merge_points

  def __call__(self, batch: List):
    assert type(batch) == list

    outputs = {}
    for key in batch[0].keys():
      outputs[key] = [b[key] for b in batch]

      # Merge a batch of octrees into one super octree
      if 'octree' in key:
        octree = merge_octrees(outputs[key])
        octree.construct_all_neigh()
        outputs[key] = octree

      if 'points' in key and self.merge_points:
        outputs[key] = merge_points(outputs[key])

      # Convert the labels to a Tensor
      if 'label' in key:
        outputs['label'] = torch.tensor(outputs[key])

    return outputs


def get_modelnet40_dataset(flags):
  transform = Transform(flags)
  read_ply = ReadPly(has_normal=True)
  collate_batch = CollateBatch()

  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_ply, in_memory=flags.in_memory,
                    take=flags.take)
  return dataset, collate_batch


def get_seg_shapenet_dataset(flags):
  transform = Transform(flags)
  read_ply = ReadPly(has_normal=True, has_label=True)
  collate_batch = CollateBatch(merge_points=True)

  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_ply, in_memory=flags.in_memory,
                    take=flags.take)
  return dataset, collate_batch
