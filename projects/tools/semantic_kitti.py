# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import zipfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_folder', type=str, default='data/SemanticKitti')
args = parser.parse_args()


def unzip_files():

  for filename in ['data_odometry_labels.zip', 'data_odometry_velodyne.zip']:
    zip_name = os.path.join(args.root_folder, filename)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
      zip_ref.extractall(args.root_folder)
      print('Unzip %s' % zip_name)


def generate_file_list():

  data_folder = 'dataset/sequences'
  root_folder = os.path.join(args.root_folder, data_folder)
  split = {
      'train': ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
      'val': ['08'],
      'test': ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
  }
  split['train_val'] = split['train'] + split['val']

  for key, value in split.items():
    filenames = []
    for seq in value:
      files = sorted(os.listdir(os.path.join(root_folder, seq, 'velodyne')))
      files = [os.path.join(data_folder, seq, 'velodyne', x) for x in files]
      filenames += files

    filelist = os.path.join(args.root_folder, key + '.txt')
    print('Create filelist: ' + filelist)
    with open(filelist, 'w') as fid:
      for filename in filenames:
        fid.write(filename + ' 0\n')


if __name__ == '__main__':
  unzip_files()
  generate_file_list()
