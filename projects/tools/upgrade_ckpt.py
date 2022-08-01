# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import sys
import torch

filename = sys.argv[1]
weights = torch.load(filename, map_location='cpu')
upgrade = dict()
for key, val in weights.items():
  print(key, val.shape)

  if 'conv.weight' in key:
    last_dim = 8 if 'downsample' in key or 'upsample' in key else 27
    upgrade[key] = val.view(val.shape[0], -1, last_dim).permute(2, 1, 0)

  elif 'conv1x1.weight' in key:
    if 'conv1x1.conv1x1' in key:
      new_key = key.replace('conv1x1.conv1x1', 'conv.linear')
    else:
      new_key = key.replace('conv1x1', 'linear')
    upgrade[new_key] = val.squeeze(2)

  elif 'conv1x1.bias' in key:
    new_key = key.replace('conv1x1.bias', 'linear.bias')
    upgrade[new_key] = val

  else:
    upgrade[key] = val

torch.save(upgrade, filename[:-4] + '.upgrade.pth')
print('succ!')
