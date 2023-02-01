# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
from thsolver import Solver

from datasets import get_modelnet40_dataset


class ClsSolver(Solver):

  def get_model(self, flags):
    if flags.name.lower() == 'lenet':
      model = ocnn.models.LeNet(
          flags.channel, flags.nout, flags.stages, flags.nempty)
    elif flags.name.lower() == 'resnet':
      model = ocnn.models.ResNet(
          flags.channel, flags.nout, flags.resblock_num, flags.stages, flags.nempty)
    elif flags.name.lower() == 'hrnet':
      model = ocnn.models.HRNet(
          flags.channel, flags.nout, flags.stages, nempty=flags.nempty)
    else:
      raise ValueError
    return model

  def get_dataset(self, flags):
    return get_modelnet40_dataset(flags)

  def get_input_feature(self, octree):
    flags = self.FLAGS.MODEL
    octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
    data = octree_feature(octree)
    return data

  def loss_function(self, logit, label):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logit, label.long())
    return loss

  def forward(self, batch):
    octree, label = batch['octree'].cuda(), batch['label'].cuda()
    data = self.get_input_feature(octree)
    logits = self.model(data, octree, octree.depth)
    loss = self.loss_function(logits, label)
    pred = torch.argmax(logits, dim=1)
    accu = pred.eq(label).float().mean()
    return loss, accu

  def train_step(self, batch):
    loss, accu = self.forward(batch)
    return {'train/loss': loss, 'train/accu': accu}

  def test_step(self, batch):
    with torch.no_grad():
      loss, accu = self.forward(batch)
    return {'test/loss': loss, 'test/accu': accu}


if __name__ == "__main__":
  ClsSolver.main()
