# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import numpy as np
import unittest

import ocnn
from .utils import get_octree, get_batch_octree


class TesOctree(unittest.TestCase):

  def init_points(self):
    points = torch.Tensor([[-1, -1, -1], [0, 0, -1], [0.0625, 0.0625, -1]])
    normals = torch.Tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
    features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
    labels = torch.Tensor([[0], [2], [2]])
    return ocnn.octree.Points(points, normals, features, labels)

  def build_octree(self, device):
    point_cloud = self.init_points().to(device)
    octree = ocnn.octree.Octree(depth=5, full_depth=1, device=device)
    octree.build_octree(point_cloud)
    octree = octree.to('cpu')

    # test node number
    nnum = torch.Tensor([1, 8, 16, 16, 16, 16])
    nnum_nempty = torch.Tensor([1, 2, 2, 2, 2, 3])
    self.assertTrue((octree.nnum == nnum).all())
    self.assertTrue((octree.nnum_nempty == nnum_nempty).all())

    # test the key
    keys = [
        torch.Tensor([0]),
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 48,
                      49, 50, 51, 52, 53, 54, 55, ]),
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 384, 385,
                      386, 387, 388, 389, 390, 391, ]),
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 3072, 3073,
                      3074, 3075, 3076, 3077, 3078, 3079, ]),
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 24576, 24577,
                      24578, 24579, 24580, 24581, 24582, 24583]), ]
    for d in range(0, 6):
      self.assertTrue((octree.keys[d] == keys[d]).all())

    # test the children
    children = [
        torch.Tensor([0]),
        torch.Tensor([0, -1, -1, -1, -1, -1, 1, -1]),
        torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1,
                      1, -1, -1, -1, -1, -1, -1, -1]),
        torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1,
                      1, -1, -1, -1, -1, -1, -1, -1]),
        torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1,
                      1, -1, -1, -1, -1, -1, -1, -1]),
        torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1,
                      1, -1, -1, -1, -1, -1, 2, -1]), ]

    for d in range(0, 6):
      self.assertTrue((octree.children[d] == children[d]).all())

    # test the signal
    normals = torch.Tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
    features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
    self.assertTrue((octree.normals[5] == normals).all())
    self.assertTrue((octree.features[5] == features).all())

  def check_octree(self, octree, data):
    # check node numbers
    self.assertTrue(
        np.array_equal(octree.nnum.numpy(), data['nnum']))
    self.assertTrue(
        np.array_equal(octree.nnum_nempty.numpy(), data['nnum_nempty']))

    # check key
    self.assertTrue(
        np.array_equal(torch.cat(octree.keys).numpy(), data['key']))
    self.assertTrue(
        np.array_equal(torch.cat(octree.children).numpy(), data['child']))

    # check normals
    normals = octree.normals[octree.depth].numpy()
    self.assertTrue(
        np.allclose(normals, data['feature'][:, :3]))

  def test_build_octree(self):
    self.build_octree('cpu')
    if torch.cuda.is_available():
      self.build_octree('cuda')

  def test_octree_with_data(self):
    for i in range(1, 6):
      octree, data = get_octree(i, return_data=True)
      self.check_octree(octree, data)

  def test_merge_octree_with_data(self):
    folder = os.path.dirname(__file__)
    data = np.load(os.path.join(folder, 'data/batch_45.npz'))

    devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
    for device in devices:
      octree = get_batch_octree(device).to('cpu')
      self.check_octree(octree, data)

      # check neigh
      self.assertTrue(np.array_equal(
          torch.cat(octree.neighs[1:], dim=0).numpy(), data['neigh']))

  def test_search_key(self):

    folder = os.path.dirname(__file__)
    data = np.load(os.path.join(folder, 'data/search_key.npz'))
    octree = get_batch_octree()

    depth = 5
    xyzb = torch.from_numpy(data['xyzb'])
    key = ocnn.octree.xyz2key(
        xyzb[:, 0], xyzb[:, 1], xyzb[:, 2], xyzb[:, 3], depth)
    idx = octree.search_key(key, depth, nempty=False)
    idx_ne = octree.search_key(key, depth, nempty=True)
    idx_xyz = octree.search_xyzb(xyzb, depth, nempty=False)
    idx_xyz_ne = octree.search_xyzb(xyzb, depth, nempty=True)

    self.assertTrue(np.array_equal(idx.numpy(), data['idx']))
    self.assertTrue(np.array_equal(idx_ne.numpy(), data['idx_ne']))
    self.assertTrue(np.array_equal(idx_xyz.numpy(), data['idx']))
    self.assertTrue(np.array_equal(idx_xyz_ne.numpy(), data['idx_ne']))

  def test_octree_grow(self):

    folder = os.path.dirname(__file__)
    data = np.load(os.path.join(folder, 'data/batch_45.npz'))
    octree_gt = get_batch_octree()

    depth = 6
    full_depth = 3
    octree = ocnn.octree.Octree(depth, full_depth, batch_size=2)
    for d in range(full_depth+1):
      octree.octree_grow_full(depth=d)

    for d in range(full_depth, depth+1):
      split = octree_gt.children[d] >= 0
      octree.octree_split(split, d)
      if d < depth:
        octree.octree_grow(d + 1)

    octree.normals[depth] = octree_gt.normals[depth]

    self.check_octree(octree, data)
    self.assertTrue(np.array_equal(
        torch.cat(octree.neighs[1:], dim=0).numpy(), data['neigh']))

  def test_octree_neigh(self):

    octree1 = get_batch_octree()
    octree2 = get_batch_octree()

    # After change the full_depth, the neigh of `octree2 in` depth 3 is
    # computed by the other if-branch in `Octree.construct_neigh()`
    octree2.full_depth = 2
    octree2.construct_neigh(depth=3)

    self.assertTrue(torch.equal(octree1.neighs[3], octree2.neighs[3]))

  def test_batch_id(self):
    # get the octree batch
    octree1 = get_octree(4)
    octree2 = get_octree(5)
    octree = ocnn.octree.merge_octrees([octree1, octree2])

    # test1
    depth = 4
    for nempty in [True, False]:
      b0 = octree.xyzb(depth, nempty)[3]
      b1 = octree.batch_id(depth, nempty)
      if nempty:
        b2 = torch.cat([torch.zeros(octree1.nnum_nempty[depth]),
                        torch.ones(octree2.nnum_nempty[depth])])
      else:
        b2 = torch.cat([torch.zeros(octree1.nnum[depth]),
                        torch.ones(octree2.nnum[depth])])
      self.assertTrue(torch.equal(b0, b1))
      self.assertTrue(torch.equal(b0, b2.long()))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
