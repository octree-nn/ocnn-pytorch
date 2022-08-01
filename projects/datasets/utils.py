# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import numpy as np
from plyfile import PlyData


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


class Transform(ocnn.dataset.Transform):
  r''' Wraps :class:`ocnn.data.Transform` for convenience.
  '''

  def __init__(self, flags):
    super().__init__(**flags)
    self.flags = flags
