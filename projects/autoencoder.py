# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import torch.nn.functional as F
import numpy as np
import ocnn

from ocnn.octree import Octree
from thsolver import Solver, get_config
from dataseto import get_ae_shapenet_dataset

# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


class AutoEncoderSolver(Solver):

  def get_model(self, flags):
    return ocnn.models.AutoEncoder(
        flags.channel, flags.nout, flags.depth, flags.full_depth,
        feature=flags.feature)

  def get_dataset(self, flags):
    return get_ae_shapenet_dataset(flags)

  def get_ground_truth_signal(self, octree):
    flags = self.FLAGS.MODEL
    octree_feature = ocnn.modules.InputFeature('ND', nempty=True)
    data = octree_feature(octree)
    return data

  def compute_loss(self, octree: ocnn.octree.Octree, model_out: dict):
    # octree splitting loss
    output = dict()
    logits = model_out['logits']
    for d in logits.keys():
      logitd = logits[d]
      label_gt = octree.nempty_mask(d).long()
      output['loss_%d' % d] = F.cross_entropy(logitd, label_gt)
      output['accu_%d' % d] = logitd.argmax(1).eq(label_gt).float().mean()

    # octree regression loss
    signal = model_out['signal']
    signal_gt = self.get_ground_truth_signal(octree)
    output['loss_reg'] = torch.mean(torch.sum((signal_gt - signal)**2, dim=1))

    # total loss
    losses = [val for key, val in output.items() if 'loss' in key]
    output['loss'] = torch.sum(torch.stack(losses))
    return output

  def model_forward(self, batch):
    octree = batch['octree'].cuda()
    model_out = self.model(octree, update_octree=False)
    output = self.compute_loss(octree, model_out)
    return output

  def train_step(self, batch):
    output = self.model_forward(batch)
    output = {'train/' + key: val for key, val in output.items()}
    return output

  def test_step(self, batch):
    output = self.model_forward(batch)
    output = {'test/' + key: val for key, val in output.items()}
    return output

  def eval_step(self, batch):
    # forward the model
    octree_in = batch['octree'].cuda(non_blocking=True)
    output = self.model(octree_in, update_octree=True)
    points_out = self.octree2pts(output['octree_out'])

    # save the output point clouds
    points_in = batch['points']
    filenames = batch['filename']
    for i, filename in enumerate(filenames):
      pos = filename.rfind('.')
      if pos != -1: filename = filename[:pos]  # remove the suffix
      filename_in = os.path.join(self.logdir, filename + '.in.xyz')
      filename_out = os.path.join(self.logdir, filename + '.out.xyz')
      os.makedirs(os.path.dirname(filename_in), exist_ok=True)

      # NOTE: it consumes much time to save point clouds to hard disks
      points_in[i].save(filename_in)
      np.savetxt(filename_out, points_out[i].cpu().numpy(), fmt='%.6f')

  def octree2pts(self, octree: Octree):
    depth = octree.depth
    batch_size = octree.batch_size

    signal = octree.features[depth]
    normal = F.normalize(signal[:, :3])
    displacement = signal[:, 3:]

    x, y, z, _ = octree.xyzb(depth, nempty=True)
    xyz = torch.stack([x, y, z], dim=1) + 0.5 + displacement * normal
    xyz = xyz / 2**(depth - 1) - 1.0  # [0, 2^depth] -> [-1, 1]
    point_cloud = torch.cat([xyz, normal], dim=1)

    batch_id = octree.batch_id(depth, nempty=True)
    points_num = [torch.sum(batch_id == i) for i in range(batch_size)]
    points = torch.split(point_cloud, points_num)
    return points


if __name__ == '__main__':
  AutoEncoderSolver.main()
