# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import json
import argparse
import wget
import zipfile
import ssl
import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=False, default='prepare_dataset',
                    help='The command to run.')

args = parser.parse_args()

# The following line is to deal with the error of "SSL: CERTIFICATE_VERIFY_FAILED"
# when using wget. (Ref: https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3)
ssl._create_default_https_context = ssl._create_unverified_context

abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(abs_path, 'data/shapenet_segmentation')
zip_name = 'shapenetcore_partanno_segmentation_benchmark_v0_normal'
txt_folder = os.path.join(root_folder, zip_name)
ply_folder = os.path.join(root_folder, 'points')

categories = ['02691156', '02773838', '02954340', '02958343',
              '03001627', '03261776', '03467517', '03624134',
              '03636649', '03642806', '03790512', '03797390',
              '03948459', '04099429', '04225987', '04379243']
names = ['Aero', 'Bag', 'Cap', 'Car',
         'Chair', 'EarPhone', 'Guitar', 'Knife',
         'Lamp', 'Laptop', 'Motor', 'Mug',
         'Pistol', 'Rocket', 'Skate', 'Table']
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
dis = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def download_shapenet_segmentation():

  # download via wget
  print('-> Download the dataset')
  if not os.path.exists(root_folder):
    os.makedirs(root_folder)
  url = 'https://shapenet.cs.stanford.edu/media/' + zip_name + '.zip'
  filename = os.path.join(root_folder, zip_name + '.zip')
  wget.download(url, filename)

  # unzip
  with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(root_folder)


def txt_to_ply():

  print('-> Convert txt files to ply files')
  for i, c in enumerate(categories):
    src_folder = os.path.join(txt_folder, c)
    des_folder = os.path.join(ply_folder, c)
    if not os.path.exists(des_folder):
      os.makedirs(des_folder)

    filenames = os.listdir(src_folder)
    for filename in filenames:
      filename_txt = os.path.join(src_folder, filename)
      filename_ply = os.path.join(des_folder, filename[:-4] + '.ply')

      raw = np.loadtxt(filename_txt)
      points = raw[:, :3]
      normals = raw[:, 3:6]
      label = raw[:, 6:] - dis[i]  # !!! NOTE: the displacement

      utils.save_points_to_ply(
          filename_ply, points, normals, labels=label, text=False)
      print('Save: ' + os.path.basename(filename_ply))


def generate_filelist():

  print('-> Generate filelists')
  list_folder = os.path.join(txt_folder, 'train_test_split')
  train_list_name = os.path.join(list_folder, 'shuffled_train_file_list.json')
  val_list_name = os.path.join(list_folder, 'shuffled_val_file_list.json')
  test_list_name = os.path.join(list_folder, 'shuffled_test_file_list.json')
  with open(train_list_name) as fid:
    train_list = json.load(fid)
  with open(val_list_name) as fid:
    val_list = json.load(fid)
  with open(test_list_name) as fid:
    test_list = json.load(fid)

  list_folder = os.path.join(root_folder, 'filelist')
  if not os.path.exists(list_folder):
    os.makedirs(list_folder)
  for i, c in enumerate(categories):
    filelist_name = os.path.join(list_folder, c + '_train_val.txt')
    filelist = ['%s.ply %d' % (line[11:], i) for line in train_list if c in line] + \
               ['%s.ply %d' % (line[11:], i) for line in val_list if c in line]
    with open(filelist_name, 'w') as fid:
      fid.write('\n'.join(filelist))

    filelist_name = os.path.join(list_folder, c + '_test.txt')
    filelist = ['%s.ply %d' % (line[11:], i) for line in test_list if c in line]
    with open(filelist_name, 'w') as fid:
      fid.write('\n'.join(filelist))


def prepare_dataset():
  download_shapenet_segmentation()
  txt_to_ply()
  generate_filelist()


if __name__ == '__main__':
  eval('%s()' % args.run)
