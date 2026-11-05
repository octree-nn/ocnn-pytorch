import unittest
from unittest.mock import patch

import torch

from ocnn.octree import Octree, octree_marching_cubes


def make_tree(volume, device='cpu'):
  tree = Octree(5, 3, batch_size=volume.shape[0], device=device)
  tree.build_octree_from_sdf(volume.to(device))
  return tree


def sphere_volume():
  axis = torch.arange(33, dtype=torch.float32)
  xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1)
  return ((xyz - xyz.new_tensor([15.2, 16.1, 14.7])).norm(dim=-1) - 8.37)[None]


class TestMarchingCubes(unittest.TestCase):

  def test_plane_at_upper_boundary_and_world_coordinates(self):
    axis = torch.arange(33, dtype=torch.float32)
    volume = (axis[:, None, None] - 31.25).expand(33, 33, 33)[None]
    tree = make_tree(volume)
    with patch.object(tree, 'to_sdf', side_effect=AssertionError('dense rebuild')):
      mesh, = octree_marching_cubes(tree, backend='torch')
    self.assertEqual(mesh.vertices.shape, (33 * 33, 3))
    self.assertEqual(mesh.faces.shape, (2 * 32 * 32, 3))
    torch.testing.assert_close(mesh.vertices[:, 0], torch.full((33 * 33,), 31.25))
    self.assertEqual(mesh.vertices[:, 1:].min().item(), 0)
    self.assertEqual(mesh.vertices[:, 1:].max().item(), 32)
    tris = mesh.vertices[mesh.faces]
    normals = torch.linalg.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    self.assertTrue((normals[:, 0] > 0).all())
    world, = octree_marching_cubes(tree, world=True, backend='torch')
    torch.testing.assert_close(world.vertices, mesh.vertices / 16 - 1)

  def test_sphere_matches_independent_dense_mc(self):
    from skimage.measure import marching_cubes
    volume = sphere_volume()
    mesh, = octree_marching_cubes(make_tree(volume), backend='torch')
    # skimage uses the last array dimension as MC x. Align its cube axes
    # before comparing triangulations; rotations can choose other diagonals.
    verts, faces, _, _ = marching_cubes(volume[0].numpy().transpose(2, 1, 0),
                                        0, method='lorensen')
    reference_vertices = torch.from_numpy(verts[:, ::-1].copy())
    distances = torch.cdist(reference_vertices.double(), mesh.vertices.double(),
                            compute_mode='donot_use_mm_for_euclid_dist')
    error, mapping = distances.min(1)
    self.assertLess(error.max().item(), 3.e-6)
    self.assertEqual(mapping.unique().numel(), mesh.vertices.shape[0])
    mapped_faces = mapping[torch.from_numpy(faces.copy()).long()]
    canonical = lambda f: sorted(tuple(t) for t in f.sort(1).values.tolist())
    self.assertEqual(canonical(mesh.faces), canonical(mapped_faces))
    edges = mesh.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2).sort(1).values
    _, counts = torch.unique(edges, dim=0, return_counts=True)
    self.assertTrue((counts == 2).all())
    tris = mesh.vertices[mesh.faces]
    normals = torch.linalg.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    outward = tris.mean(1) - torch.tensor([15.2, 16.1, 14.7])
    self.assertTrue(((normals * outward).sum(1) > 0).all())

  def test_exact_zero_matches_octree_partition(self):
    axis = torch.arange(33, dtype=torch.float32)
    # At x=16, the zero belongs to the inside. This must produce one plane,
    # even though the neighboring negative+zero cells are not in the tree.
    for sign in (1, -1):
      volume = (sign * (axis[:, None, None] - 16)).expand(33, 33, 33)[None]
      mesh, = octree_marching_cubes(make_tree(volume), backend='torch')
      self.assertEqual(mesh.faces.shape[0], 2048)
      self.assertEqual(mesh.vertices.shape[0], 1089)
      self.assertTrue((mesh.vertices[:, 0] == 16).all())

  def test_empty_and_batch_isolation(self):
    volume = sphere_volume()
    other = volume + 2.25
    together = make_tree(torch.cat((volume, other, torch.ones_like(volume)), 0))
    meshes = octree_marching_cubes(together, backend='torch')
    self.assertEqual(len(meshes), 3)
    for index, field in enumerate((volume, other)):
      separate, = octree_marching_cubes(make_tree(field), backend='torch')
      torch.testing.assert_close(meshes[index].vertices, separate.vertices)
      torch.testing.assert_close(meshes[index].faces, separate.faces)
    self.assertEqual(meshes[2].vertices.shape, (0, 3))
    self.assertEqual(meshes[2].faces.shape, (0, 3))
    for value in (-1., 0., 1.):
      mesh, = octree_marching_cubes(make_tree(torch.full_like(volume, value)))
      self.assertEqual(mesh.vertices.shape[0], 0)
      self.assertEqual(mesh.faces.shape[0], 0)

  def test_unsupported_inputs(self):
    tree = make_tree(sphere_volume())
    with self.assertRaisesRegex(ValueError, 'backend'):
      octree_marching_cubes(tree, backend='unknown')
    with self.assertRaisesRegex(ValueError, 'CUDA'):
      octree_marching_cubes(tree, backend='triton')
    tree.fields[5] = tree.fields[5].to(torch.int16)
    with self.assertRaisesRegex(ValueError, 'float32'):
      octree_marching_cubes(tree)
    tree = make_tree(sphere_volume())
    keep = {d: torch.ones(int(tree.nnum_nempty[d]), dtype=torch.bool)
            for d in range(3, 5)}
    # Even pruning with an all-keep mask marks the topology as pruned.
    tree.prune(keep, 3)
    with self.assertRaisesRegex(ValueError, 'pruned'):
      octree_marching_cubes(tree)

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
  def test_cuda_backends_match_cpu(self):
    axis = torch.arange(33, dtype=torch.float32)
    zero_plane = (axis[:, None, None] - 16).expand(33, 33, 33)[None]
    volume = torch.cat((sphere_volume(), zero_plane, torch.ones_like(zero_plane)))
    cpu = octree_marching_cubes(make_tree(volume), backend='torch')
    tree = make_tree(volume, 'cuda')
    for backend in ('torch', 'triton', 'auto'):
      with self.subTest(backend=backend):
        gpu = octree_marching_cubes(tree, backend=backend)
        for actual, expected in zip(gpu, cpu):
          torch.testing.assert_close(actual.vertices.cpu(), expected.vertices,
                                     atol=2.e-6, rtol=0)
          torch.testing.assert_close(actual.faces.cpu(), expected.faces)


if __name__ == '__main__':
  unittest.main()
