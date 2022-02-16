import os
import torch
import numpy as np
import unittest

import ocnn


class TesOctree(unittest.TestCase):

  def init_points(self):
    points = torch.Tensor([[-1, -1, -1], [0, 0, -1], [0.0625, 0.0625, -1]])
    normals = torch.Tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
    features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
    labels = torch.Tensor([[0], [2], [2]])
    return ocnn.octree.Points(points, normals, features, labels)

  def test_build_octree(self):
    point_cloud = self.init_points()
    octree = ocnn.octree.Octree(depth=5, full_depth=1)
    octree.build_octree(point_cloud)

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

    #  test the signal
    normals = torch.Tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
    features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
    self.assertTrue((octree.normals[5] == normals).all())
    self.assertTrue((octree.features[5] == features).all())

  def build_octree(self, data):
    points, normals = data['points'], data['normals']
    point_cloud = ocnn.octree.Points(
        torch.from_numpy(points), torch.from_numpy(normals))

    octree = ocnn.octree.Octree(
        data['depth'].item(), full_depth=data['full_depth'].item())
    octree.build_octree(point_cloud)
    return octree

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
        np.array_equal(normals, data['feature'][:, :3]))

  def test_octree_with_data(self):
    folder = os.path.dirname(__file__)
    data_folder = os.path.join(folder, 'data/octree')
    filenames = os.listdir(data_folder)
    for filename in filenames:
      data = np.load(os.path.join(data_folder, filename))
      octree = self.build_octree(data)
      self.check_octree(octree, data)

  def test_merge_octree_with_data(self):
    folder = os.path.dirname(__file__)
    data_folder = os.path.join(folder, 'data/octree')

    data1 = np.load(os.path.join(data_folder, 'test_004.npz'))
    data2 = np.load(os.path.join(data_folder, 'test_005.npz'))
    octree1 = self.build_octree(data1)
    octree2 = self.build_octree(data2)

    octree = ocnn.octree.Octree(
        data1['depth'].item(), full_depth=data1['full_depth'].item())
    octree.merge_octrees([octree1, octree2])

    data = np.load(os.path.join(folder, 'data/batch_45.npz'))
    self.check_octree(octree, data)

    # check neigh
    octree.construct_all_neigh()
    self.assertTrue(
        np.array_equal(torch.cat(octree.neighs[1:], dim=0).numpy(), data['neigh']))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
