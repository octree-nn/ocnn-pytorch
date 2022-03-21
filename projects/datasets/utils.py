import torch
import numpy as np
from plyfile import PlyData
from ocnn.octree import Points, Octree


class ReadPly:

  def __init__(self, has_normal: bool = True, has_color: bool = False,
               has_label: bool = False):
    self.has_normal = has_normal
    self.has_color = has_color
    self.has_label = has_label

  def __call__(self, filename: str):
    plydata = PlyData.read(filename)
    vtx = plydata['vertex']

    output = dict()
    points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)
    output['points'] = points.astype(np.float32)
    if self.has_normal:
      normal = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
      output['normals'] = normal.astype(np.float32)
    if self.has_color:
      color = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=1)
      output['colors'] = color.astype(np.float32)
    if self.has_label:
      label = vtx['label']
      output['labels'] = label.astype(np.int32)

    return output


class Transform:

  def __init__(self, flags):
    self.flags = flags

  def __call__(self, sample: dict, idx: int):

    points = self.preprocess(sample, idx)
    output = self.transform(points, idx)
    return output

  def preprocess(self, sample: dict, idx: int):

    xyz = torch.from_numpy(sample['points'])
    normal = torch.from_numpy(sample['normal'])
    points = Points(xyz, normal)
    return points

  def transform(self, points: Points, idx: int):

    # Apply the general transformations provided by ocnn.
    # The augmentations including rotation, scaling, and jittering.
    if self.flags.distort:
      rng_angle, rng_scale, rng_jitter = self.rnd_parameters()
      points.rotate(rng_angle)
      points.scale(rng_scale)
      points.translate(rng_jitter)

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
