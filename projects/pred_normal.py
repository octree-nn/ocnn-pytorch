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
from thsolver import Solver
from datasets import get_shapenet_dataset

# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


class NormalEstimationSolver(Solver):

  def get_model(self, flags):
    return ocnn.models.UNet(flags.channel, flags.channel_out, nempty=False)

  def get_dataset(self, flags):
    return get_shapenet_dataset(flags)

  def compute_loss(self, octree: Octree, model_out: dict):
    # octree splitting loss
    output = dict()
    depth = octree.depth
    logit = model_out[:, 4:].squeeze(1)
    label_gt = octree.nempty_mask(depth).float()
    output['loss_%d' % depth] = F.binary_cross_entropy_with_logits(logit, label_gt)
    output['accu_%d' % depth] = (logit.sigmoid().round() == label_gt).float().mean()

    # octree regression loss
    signal = model_out[:, :4][label_gt.bool()]
    signal_gt = octree.get_input_feature('ND', nempty=True)
    output['loss_reg'] = torch.mean(torch.sum((signal_gt - signal)**2, dim=1))

    # total loss
    losses = [val for key, val in output.items() if 'loss' in key]
    output['loss'] = torch.sum(torch.stack(losses))
    return output

  def model_forward(self, batch):
    octree = batch['octree'].cuda()
    data = torch.ones(octree.nnum[octree.depth], 1, device=octree.device)
    model_out = self.model(data, octree, octree.depth)
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
    self.model.train()
    octree = batch['octree'].cuda(non_blocking=True)
    data = torch.ones(octree.nnum[octree.depth], 1, device=octree.device)
    model_out = self.model(data, octree, octree.depth)
    points_out = self.octree2pts(model_out, octree)

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

  def octree2pts(self,  model_out, octree: Octree):
    depth = octree.depth
    batch_size = octree.batch_size

    label = model_out[:, 4:].squeeze().sigmoid().round()
    octree.octree_split(label, depth)

    model_out = model_out[:, :4][label.bool()]
    normal = F.normalize(model_out[:, :3])
    displacement = model_out[:, 3:4]

    x, y, z, _ = octree.xyzb(depth, nempty=True)
    xyz = torch.stack([x, y, z], dim=1) + 0.5 + displacement * normal
    xyz = xyz / 2**(depth - 1) - 1.0  # [0, 2^depth] -> [-1, 1]
    point_cloud = torch.cat([xyz, normal], dim=1)

    batch_id = octree.batch_id(depth, nempty=True)
    points_num = [torch.sum(batch_id == i) for i in range(batch_size)]
    points = torch.split(point_cloud, points_num)
    return points


if __name__ == '__main__':
  NormalEstimationSolver.main()
