import os
import math
import wget
import zipfile
import trimesh
import argparse
import trimesh.sample
import numpy as np

import utils


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=False, default='prepare_dataset',
                    help='The command to run.')
parser.add_argument('--sample_num', type=int, default=5000,
                    help='The sample number')
parser.add_argument('--align_y', type=str, required=False, default='false',
                    help='Align the points with y axis')

args = parser.parse_args()
abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(abs_path, 'data/ModelNet40')


def download_m40():

  # download via wget
  print('-> Download the dataset.')
  if not os.path.exists(root_folder):
    os.makedirs(root_folder)
  url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
  filename = os.path.join(root_folder, 'ModelNet40.zip')
  wget.download(url, filename)

  # unzip
  with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(root_folder)


def _clean_off_file(filename):
  # read the contents of the file
  with open(filename) as fid:
    file_str = fid.read()
  # fix the file
  if file_str[0:3] != 'OFF':
    print('Error: not an OFF file: ' + filename)
  elif file_str[0:4] != 'OFF\n':
    print('Info: fix an OFF file: ' + filename)
    new_str = file_str[0:3] + '\n' + file_str[3:]
    with open(filename, 'w') as f_rewrite:
      f_rewrite.write(new_str)


def _get_point_folder():
  sample_num = args.sample_num
  align_y = args.align_y.lower() == 'true'
  suffix = '.ply.{}k'.format(sample_num // 1000) + ('.y' if align_y else '')
  points_folder = os.path.join(root_folder, 'ModelNet40' + suffix)
  return points_folder


def get_filelist(root_folder, train=True, suffix='off', ratio=1.0):
  filelist, category = [], []
  folders = sorted(os.listdir(root_folder))
  assert(len(folders) == 40)
  for idx, folder in enumerate(folders):
    subfolder = 'train' if train else 'test'
    current_folder = os.path.join(root_folder, folder, subfolder)
    filenames = sorted(os.listdir(current_folder))
    filenames = [fname for fname in filenames if fname.endswith(suffix)]
    total_num = math.ceil(len(filenames) * ratio)
    for i in range(total_num):
      filelist.append(os.path.join(folder, subfolder, filenames[i]))
      category.append(idx)
  return filelist, category


def move_files(src_folder, des_folder, suffix):
  folders = os.listdir(src_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_src_folder = os.path.join(src_folder, folder, subfolder)
      curr_des_folder = os.path.join(des_folder, folder, subfolder)
      if not os.path.exists(curr_des_folder):
        os.makedirs(curr_des_folder)
      filenames = os.listdir(curr_src_folder)
      for filename in filenames:
        if filename.endswith(suffix):
          os.rename(os.path.join(curr_src_folder, filename),
                    os.path.join(curr_des_folder, filename))


def convert_mesh_to_points():

  print('-> Sample points on meshes')
  # Delete the following 3 files from training set since the scale of these
  # meshes is too large and the virtualscanner can not deal with them.
  mesh_folder = os.path.join(root_folder, 'ModelNet40')
  filelist = ['cone/train/cone_0117.off',
              'curtain/train/curtain_0066.off',
              'car/train/car_0021.off.off']
  for filename in filelist:
    filename = os.path.join(mesh_folder, filename)
    if os.path.exists(filename):
      os.remove(filename)

  # clean the off files
  train_list, _ = get_filelist(mesh_folder, train=True, suffix='off')
  test_list, _ = get_filelist(mesh_folder, train=False, suffix='off')
  filelist = train_list + test_list
  for filename in filelist:
    _clean_off_file(os.path.join(mesh_folder, filename))

  # run mesh sampling
  sample_num = args.sample_num
  ply_folder = _get_point_folder()
  for filename in filelist:
    filename_off = os.path.join(mesh_folder, filename)
    mesh = trimesh.load(filename_off, force='mesh')

    # transform to align y
    if args.align_y.lower() == 'true':
      mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
      mesh.vertices = mesh.vertices @ mat

    points, idx = trimesh.sample.sample_surface(mesh, sample_num)
    normals = mesh.face_normals[idx]

    # save to disk
    filename_ply = os.path.join(ply_folder, filename[:-3] + 'ply')
    utils.save_points_to_ply(filename_ply, points, normals)
    print('Save:', filename_ply)


def generate_points_filelist():

  print('-> Generate filelists')
  points_folder = _get_point_folder()
  list_folder = os.path.join(root_folder, 'filelist')
  if not os.path.exists(list_folder):
    os.makedirs(list_folder)

  for folder in ['train', 'test']:
    train = folder == 'train'
    filelist, idx = get_filelist(points_folder, train=train, suffix='ply')
    filename = os.path.join(list_folder, 'm40_%s.txt' % folder)
    print('Save to %s' % filename)
    with open(filename, 'w') as fid:
      for i in range(len(filelist)):
        fid.write('%s %d\n' % (filelist[i], idx[i]))

  ratios = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
  for folder in ['train', 'test']:
    train = folder == 'train'
    for ratio in ratios:
      if train == False and ratio < 1: continue
      filename = os.path.join(list_folder, 'm40_%.02f_%s.txt' % (ratio, folder))
      filelist, idx = get_filelist(points_folder, train=train,
                                   suffix='ply', ratio=ratio)
      print('Save to %s' % filename)
      with open(filename, 'w') as fid:
        for i in range(len(filelist)):
          fid.write('%s %d\n' % (filelist[i], idx[i]))


def prepare_dataset():
  download_m40()
  convert_mesh_to_points()
  generate_points_filelist()


if __name__ == '__main__':
  eval('%s()' % args.run)
