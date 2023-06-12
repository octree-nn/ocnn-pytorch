# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=False, default='train')
parser.add_argument('--alias', type=str, required=False, default='scannet')
parser.add_argument('--gpu', type=str, required=False, default='0')
parser.add_argument('--port', type=str, required=False, default='10001')
parser.add_argument('--ckpt', type=str, required=False, default='\'\'')
args = parser.parse_args()


def execute_command(cmds):
  cmd = ' '.join(cmds)
  print('Execute: \n' + cmd + '\n')
  os.system(cmd)


def train():
  cmds = [
      'python segmentation.py',
      '--config configs/seg_scannet.yaml',
      'SOLVER.gpu  {},'.format(args.gpu),
      'SOLVER.alias  {}'.format(args.alias), 
      'SOLVER.dist_url tcp://localhost:{}'.format(args.port), ]
  execute_command(cmds)


def train_all():
  cmds = [
      'python segmentation.py',
      '--config configs/seg_scannet.yaml',
      'SOLVER.gpu  {},'.format(args.gpu),
      'SOLVER.alias  {}'.format(args.alias),
      'SOLVER.dist_url tcp://localhost:{}'.format(args.port), 
      'DATA.train.filelist data/scannet/scannetv2_train_val.txt', ]
  execute_command(cmds)


def test():
  # get the predicted probabilities for each point
  ckpt = 'logs/scannet/D10_2cm_{}/checkpoints/00600.model.pth'.format(args.alias)
  if args.ckpt != '\'\'': ckpt = args.ckpt   # use args.ckpt if provided
  cmds = [
      'python segmentation.py',
      '--config configs/seg_scannet_eval.yaml',
      'SOLVER.gpu  {},'.format(args.gpu),
      'SOLVER.eval_epoch 72',  # voting with 72 predictions
      'SOLVER.alias test_{}'.format(args.alias),
      'SOLVER.ckpt {}'.format(ckpt),
      'DATA.test.distort True', ]
  execute_command(cmds)

  # map the probabilities to labels
  cmds = [
      'python tools/seg_scannet.py',
      '--run generate_output_seg',
      '--path_in data/scannet/test',
      '--path_pred logs/scannet/D10_2cm_eval_test_{}'.format(args.alias),
      '--path_out logs/scannet/D10_2cm_eval_test_seg_{}'.format(args.alias),
      '--filelist  data/scannet/scannetv2_test.txt', ]
  execute_command(cmds)


def validate():
  # get the predicted probabilities for each point
  ckpt = 'logs/scannet/D10_2cm_{}/checkpoints/00600.model.pth'.format(args.alias)
  if args.ckpt != '\'\'': ckpt = args.ckpt   # use args.ckpt if provided
  cmds = [
      'python segmentation.py',
      '--config configs/seg_scannet_eval.yaml',
      'SOLVER.gpu  {},'.format(args.gpu),
      'SOLVER.eval_epoch 12',  # voting with 12 predictions
      'SOLVER.alias val_{}'.format(args.alias),
      'SOLVER.ckpt {}'.format(ckpt),
      'DATA.test.distort True',
      'DATA.test.location  data/scannet/train',
      'DATA.test.filelist data/scannet/scannetv2_val.txt', ]
  execute_command(cmds)

  # map the probabilities to labels
  cmds = [
      'python tools/seg_scannet.py',
      '--run generate_output_seg',
      '--path_in data/scannet/train',
      '--path_pred logs/scannet/D10_2cm_eval_val_{}'.format(args.alias),
      '--path_out logs/scannet/D10_2cm_eval_val_seg_{}'.format(args.alias),
      '--filelist  data/scannet/scannetv2_val.txt', ]
  execute_command(cmds)

  # calculate the mIoU
  cmds = [
      'python tools/seg_scannet.py',
      '--run calc_iou',
      '--path_in data/scannet/train',
      '--path_pred logs/scannet/D10_2cm_eval_val_seg_{}'.format(args.alias), ]
  execute_command(cmds)


if __name__ == '__main__':
  eval('%s()' % args.run)
