# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import argparse
import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True,
                    help='The path of the ckpt file')
parser.add_argument('--skips', type=str, required=False, nargs='*',
                    help="Skip specific variables")
args = parser.parse_args()

ckpt = args.ckpt
skips = args.skips if args.skips else []


def list_var():
  # all_vars = torch.load(ckpt, map_location='cpu')['model']
  all_vars = torch.load(ckpt, map_location='cpu')
  total_num, counter = 0, 0
  for name, var in all_vars.items():
    shape = var.shape

    exclude = False
    for s in skips:
      exclude = s in name
      if exclude:
        break
    if exclude:
      continue

    shape_str = '; '.join([str(s) for s in shape])
    shape_num = np.prod(shape)
    print("{}, {}, [{}], {}".format(counter, name, shape_str, shape_num))
    total_num += shape_num
    counter += 1

  print('Total parameters: {}'.format(total_num))


if __name__ == '__main__':
  list_var()
