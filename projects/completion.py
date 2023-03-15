# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import ocnn
import numpy as np

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
    octree_gt = batch['octree_gt'].cuda(non_blocking=True)
    octree_in = batch['octree_in'].cuda(non_blocking=True)
    model_out = self.model(octree_in, octree_gt, update_octree=False)
    output = self.compute_loss(octree_gt, model_out)
    return output

  def eval_step(self, batch):
    # forward the model
    octree_in = batch['octree_in'].cuda(non_blocking=True)
    output = self.model(octree_in, update_octree=True)
    points_out = self.octree2pts(output['octree_out'])

    # save the output point clouds
    # NOTE: Curretnly, it consumes much time to save point clouds to hard disks
    points_in = batch['points_in']
    filenames = batch['filename']
    for i, filename in enumerate(filenames):
      pos = filename.rfind('.')
      if pos != -1: filename = filename[:pos]  # remove the suffix
      filename_in = os.path.join(self.logdir, filename + '.in.xyz')
      filename_out = os.path.join(self.logdir, filename + '.out.xyz')

      folder = os.path.dirname(filename_in)
      if not os.path.exists(folder): os.makedirs(folder)

      points_in[i].save(filename_in)
      np.savetxt(filename_out, points_out[i].cpu().numpy(), fmt='%.6f')


if __name__ == '__main__':
  CompletionSolver.main()
