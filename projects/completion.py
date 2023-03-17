# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn

from datasets import get_completion_dataset
from autoencoder import AutoEncoderSolver


class CompletionSolver(AutoEncoderSolver):

  def get_model(self, flags):
    return ocnn.models.OUNet(
        flags.channel, flags.nout, flags.depth, flags.full_depth,
        feature=flags.feature)

  def get_dataset(self, flags):
    return get_completion_dataset(flags)

  def model_forward(self, batch):
    octree_in = batch['octree'].cuda(non_blocking=True)
    octree_gt = batch['octree_gt'].cuda(non_blocking=True)
    model_out = self.model(octree_in, octree_gt, update_octree=False)
    output = self.compute_loss(octree_gt, model_out)
    return output


if __name__ == '__main__':
  CompletionSolver.main()
