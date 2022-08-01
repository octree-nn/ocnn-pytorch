# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import sys

root_folder = sys.argv[1]

folders = os.listdir(root_folder)
for folder in folders:  # categories
  curr_folder = os.path.join(root_folder, folder)
  if not os.path.isdir(curr_folder): continue

  sub_folders = os.listdir(curr_folder)
  for sub_folder in sub_folders:  # ratios
    curr_subfolder = os.path.join(curr_folder, sub_folder, 'checkpoints')
    if not os.path.isdir(curr_subfolder): continue

    # only keep the second last file
    ckpts = sorted(os.listdir(curr_subfolder))
    if len(ckpts) == 1: continue
    for ckpt in ckpts[:-2] + ckpts[-1:]:
      filename = os.path.join(curr_subfolder, ckpt)
      print('Remove ' + filename)
      os.remove(filename)
