import os
import torch
import torch.functional as F
import numpy as np
from tqdm import tqdm

import ocnn
from solver import Solver, get_config
from datasets import get_ae_shapenet_dataset


class AutoEncoderSolver(Solver):

  def get_model(self, flags):
    return ocnn.models.AutoEncoder(
        flags.channel_in, flags.nout, flags.depth, flags.full_depth,
        feature=flags.feature)

  def get_dataset(self, flags):
    return get_ae_shapenet_dataset(flags)

  def get_ground_truth_signal(self, octree):
    flags = self.FLAGS.MODEL
    octree_feature = ocnn.modules.InputFeature(flags.feature, nempty=True)
    data = octree_feature(octree)
    return data

  def compute_loss(self, octree, model_out):
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
    octree = batch['octree'].cuda()
    output = self.model(octree, update_octree=True)

    # # extract the output point cloud
    # filename = batch['filename'][0]
    # pos = filename.rfind('.')
    # if pos != -1: filename = filename[:pos]  # remove the suffix
    # filename = os.path.join(self.logdir, filename + '.obj')
    # folder = os.path.dirname(filename)
    # if not os.path.exists(folder): os.makedirs(folder)
    # bbox = batch['bbox'][0].numpy() if 'bbox' in batch else None
    # self.extract_mesh(output['neural_mpu'], filename, bbox)

    # # save the input point cloud
    # filename = filename[:-4] + '.input.ply'

  @classmethod
  def update_configs(cls):
    FLAGS = get_config()

    FLAGS.DATA.train.points_scale = 128
    FLAGS.DATA.test = FLAGS.DATA.train.clone()

    FLAGS.MODEL.depth = 6
    FLAGS.MODEL.full_depth = 2


if __name__ == '__main__':
  AutoEncoderSolver.main()
