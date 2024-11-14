# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import argparse
import wget
import zipfile
import ssl
import numpy as np
from tqdm import tqdm
from plyfile import PlyData


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='prepare_dataset')
args = parser.parse_args()

abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(abs_path, 'data/ae_shapenet')


# The following line is to deal with the error of "SSL: CERTIFICATE_VERIFY_FAILED"
# when using wget. (Ref: https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3)
ssl._create_default_https_context = ssl._create_unverified_context


def download_point_clouds():
  # download via wget
  if not os.path.exists(root_folder):
    os.makedirs(root_folder)
  url = 'https://www.dropbox.com/s/z2x0mw4ai18f855/ocnn_completion.zip?dl=1'
  filename = os.path.join(root_folder, 'ae_shapenet.zip')
  if not os.path.exists(filename):
    print('Download %s' % filename)
    wget.download(url, filename)

  # unzip
  print('Unzip %s' % filename)
  with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(root_folder)


def _read_ply(filename: str):
  plydata = PlyData.read(filename)
  vtx = plydata['vertex']
  points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)
  normals = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
  return points.astype(np.float32), normals.astype(np.float32)


def _convert_ply_to_numpy(prefix='shape'):
  ply_folder = os.path.join(root_folder, prefix + '.ply')
  npy_folder = os.path.join(root_folder, prefix + '.npz')

  folders = os.listdir(ply_folder)
  for folder in folders:
    src_folder = os.path.join(ply_folder, folder)
    des_folder = os.path.join(npy_folder, folder)
    if not os.path.exists(des_folder):
      os.makedirs(des_folder)

    filenames = os.listdir(src_folder)
    for filename in tqdm(filenames):
      if filename.endswith('.ply'):
        ply_filename = os.path.join(src_folder, filename)
        npz_filename = os.path.join(des_folder, filename[:-4] + '.npz')

        # load ply
        points, normals = _read_ply(ply_filename)

        # save npy
        np.savez(npz_filename, points=points, normals=normals)


def convert_ply_to_numpy():
  _convert_ply_to_numpy('shape')
  _convert_ply_to_numpy('test.scans')


def _create_filelist(filename: str, suffix: str = '.npz'):
  filename = os.path.join(root_folder, filename)
  with open(filename, 'r') as fid:
    lines = fid.readlines()

  lines = [line.replace('.points', suffix) for line in lines]

  filename_out = filename[:-4] + suffix + '.txt'
  with open(filename_out, 'w') as fid:
    fid.write(''.join(lines))


def create_filelist():
  for filename in ['filelist_test', 'filelist_train', 'filelist_test_scans']:
    _create_filelist(filename + '.txt', suffix='.npz')
    _create_filelist(filename + '.txt', suffix='.ply')


def prepare_dataset():
  download_point_clouds()
  convert_ply_to_numpy()
  create_filelist()


if __name__ == '__main__':
  eval('%s()' % args.run)
