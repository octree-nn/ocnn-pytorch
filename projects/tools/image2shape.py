import os
import trimesh
import argparse
import numpy as np
import trimesh.sample
from tqdm import tqdm
from plyfile import PlyData

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=True,
                    help='The command to run.')
parser.add_argument('--sample_num', type=int, default=30000,
                    help='The number of points to sample.')


args = parser.parse_args()
abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(abs_path, 'data/ShapeNetV1')
categories = ['02691156', '02828884', '02933112', '02958343', '03001627',
              '03211117', '03636649', '03691459', '04090263', '04256520',
              '04379243', '04401088', '04530566']


def mesh2points():
  print('-> Sample points on meshes.')
  mesh_folder = os.path.join(root_folder, 'shapenet.mesh')
  pts_folder = os.path.join(root_folder, 'points.ply')

  # run mesh sampling
  sample_num = args.sample_num
  for category in categories:
    print('Processing ' + category)
    mesh_path = os.path.join(mesh_folder, category)
    pts_path = os.path.join(pts_folder, category)
    os.makedirs(pts_path, exist_ok=True)
    filenames = sorted(os.listdir(mesh_path))
    for filename in tqdm(filenames, ncols=80):
      filename_obj = os.path.join(mesh_path, filename)
      mesh = trimesh.load(filename_obj, force='mesh')

      # sample points
      points, idx = trimesh.sample.sample_surface(mesh, sample_num)
      normals = mesh.face_normals[idx]

      # save to disk
      filename_ply = os.path.join(pts_path, filename[:-3] + 'ply')
      utils.save_points_to_ply(filename_ply, points, normals)


def ply2npz():
  print('-> Sample points on meshes.')
  ply_folder = os.path.join(root_folder, 'points.ply')
  npz_folder = os.path.join(root_folder, 'points.npz')

  for category in categories:
    print('Processing ' + category)
    ply_path = os.path.join(ply_folder, category)
    npz_path = os.path.join(npz_folder, category)
    os.makedirs(npz_path, exist_ok=True)
    filenames = sorted(os.listdir(ply_path))
    for filename in tqdm(filenames, ncols=80):
      filename_ply = os.path.join(ply_path, filename)
      points, normals = read_ply(filename_ply)
      filename_npz = os.path.join(npz_path, filename[:-3] + 'npz')
      np.savez(filename_npz, points=points.astype(np.float16),
               normals=normals.astype(np.float16))


def read_ply(filename: str):
  plydata = PlyData.read(filename)
  vtx = plydata['vertex']
  points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)
  normals = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
  return points, normals


def create_filelist():
  pts_folder = os.path.join(root_folder, 'points.ply')
  img_folder = os.path.join(root_folder, 'renderings')
  filelist_folder = os.path.join(root_folder, 'filelist')
  os.makedirs(filelist_folder, exist_ok=True)

  filelist_train, filelist_test = [], []
  for category in categories:
    img_path = os.path.join(img_folder, category)
    pts_path = os.path.join(pts_folder, category)
    img_filenames = sorted(os.listdir(img_path))  # sort the filenames
    pts_filenames = sorted(os.listdir(pts_path))  # sort the filenames

    # the intersection of the two list
    filenames = ['%s/%s' % (category, f)
                 for f in img_filenames if f + '.ply' in pts_filenames]

    # construct the filelists
    num = int(len(filenames) * 0.8)
    filelist_train += filenames[:num]
    filelist_test += filenames[num:]

  # save to disk
  name = os.path.join(filelist_folder, 'image2shape.train.txt')
  with open(name, 'w') as fid:
    buffer = '\n'.join(filelist_train)
    fid.write(buffer)
  name = os.path.join(filelist_folder, 'image2shape.test.txt')
  with open(name, 'w') as fid:
    buffer = '\n'.join(filelist_test)
    fid.write(buffer)


if __name__ == '__main__':
  eval('%s()' % args.run)
