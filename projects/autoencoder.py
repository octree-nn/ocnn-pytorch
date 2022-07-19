import os
import torch
import numpy as np

import builder
import utils
from solver import Solver, get_config


class DualOcnnSolver(Solver):

  def get_model(self, flags):
    return builder.get_model(flags)

  def get_dataset(self, flags):
    return builder.get_dataset(flags)

  def batch_to_cuda(self, batch):
    keys = ['octree_in', 'octree_gt', 'pos', 'sdf', 'grad', 'weight', 'occu']
    for key in keys:
      if key in batch:
        batch[key] = batch[key].cuda()

  def compute_loss(self, batch, model_out):
    flags = self.FLAGS.LOSS
    loss_func = builder.get_loss_function(flags)
    output = loss_func(batch, model_out, flags.loss_type)
    return output

  def model_forward(self, batch):
    self.batch_to_cuda(batch)
    model_out = self.model(batch['octree_in'], batch['octree_gt'], batch['pos'])

    output = self.compute_loss(batch, model_out)
    losses = [val for key, val in output.items() if 'loss' in key]
    output['loss'] = torch.sum(torch.stack(losses))
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
    output = self.model.forward(batch['octree_in'].cuda())

    # extract the mesh
    filename = batch['filename'][0]
    pos = filename.rfind('.')
    if pos != -1: filename = filename[:pos]  # remove the suffix
    filename = os.path.join(self.logdir, filename + '.obj')
    folder = os.path.dirname(filename)
    if not os.path.exists(folder): os.makedirs(folder)
    bbox = batch['bbox'][0].numpy() if 'bbox' in batch else None
    self.extract_mesh(output['neural_mpu'], filename, bbox)

    # save the input point cloud
    filename = filename[:-4] + '.input.ply'
    utils.points2ply(filename, batch['points_in'][0].cpu(),
                     self.FLAGS.DATA.test.point_scale)

  @classmethod
  def update_configs(cls):
    FLAGS = get_config()
    FLAGS.SOLVER.resolution = 128       # the resolution used for marching cubes
    FLAGS.SOLVER.save_sdf = False       # save the sdfs in evaluation
    FLAGS.SOLVER.sdf_scale = 0.9        # the scale of sdfs

    FLAGS.DATA.train.point_scale = 0.5  # the scale of point clouds
    FLAGS.DATA.train.load_sdf = True    # load sdf samples
    FLAGS.DATA.train.load_occu = False  # load occupancy samples
    FLAGS.DATA.train.point_sample_num = 10000
    FLAGS.DATA.train.sample_surf_points = False

    # FLAGS.MODEL.skip_connections = True
    FLAGS.DATA.test = FLAGS.DATA.train.clone()
    FLAGS.LOSS.loss_type = 'sdf_reg_loss'


if __name__ == '__main__':
  DualOcnnSolver.main()
