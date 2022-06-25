import os
import torch
import numpy as np
from tqdm import tqdm

# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')

import ocnn
from solver import Solver, get_config
from datasets import (get_shapenet_seg_dataset, get_scannet_dataset,
                      get_kitti_dataset)


class SegSolver(Solver):

  def get_model(self, flags):
    if flags.name.lower() == 'segnet':
      model = ocnn.models.SegNet(
          flags.channel, flags.nout, flags.stages, flags.interp, flags.nempty)
    elif flags.name.lower() == 'unet':
      model = ocnn.models.UNet(
          flags.channel, flags.nout, flags.interp, flags.nempty)
    else:
      raise ValueError
    return model

  def get_dataset(self, flags):
    if flags.name.lower() == 'shapenet':
      return get_shapenet_seg_dataset(flags)
    elif flags.name.lower() == 'scannet':
      return get_scannet_dataset(flags)
    elif flags.name.lower() == 'kitti':
      return get_kitti_dataset(flags)
    else:
      raise ValueError

  def get_input_feature(self, octree):
    flags = self.FLAGS.MODEL
    octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
    data = octree_feature(octree)
    return data

  def model_forward(self, batch):
    octree = batch['octree'].cuda()
    points = batch['points'].cuda()
    data = self.get_input_feature(octree)
    query_pts = torch.cat([points.points, points.batch_id], dim=1)

    logit = self.model(data, octree, octree.depth, query_pts)
    label_mask = points.labels > self.FLAGS.LOSS.mask  # filter labels
    return logit[label_mask], points.labels[label_mask]

  def train_step(self, batch):
    logit, label = self.model_forward(batch)
    loss = self.loss_function(logit, label)
    return {'train/loss': loss}

  def test_step(self, batch):
    with torch.no_grad():
      logit, label = self.model_forward(batch)
    loss = self.loss_function(logit, label)
    accu = self.accuracy(logit, label)
    num_class = self.FLAGS.LOSS.num_class
    IoU, insc, union = self.IoU_per_shape(logit, label, num_class)

    names = ['test/loss', 'test/accu', 'test/mIoU'] + \
            ['test/intsc_%d' % i for i in range(num_class)] + \
            ['test/union_%d' % i for i in range(num_class)]
    tensors = [loss, accu, IoU] + insc + union
    return dict(zip(names, tensors))

  def eval_step(self, batch):
    with torch.no_grad():
      logit, _ = self.model_forward(batch)
    prob = torch.nn.functional.softmax(logit, dim=1)

    # The point cloud may be clipped when doing data augmentation. The
    # `inbox_mask` indicates which points are clipped. The `prob_all_pts`
    # contains the prediction for all points.
    inbox_mask = batch['inbox_mask'][0]
    assert len(batch['inbox_mask']) == 1, 'The batch_size must be 1'
    prob_all_pts = torch.zeros([inbox_mask.shape[0], prob.shape[1]])
    prob_all_pts[inbox_mask] = prob.cpu()

    # Aggregate predictions across different epochs
    filename = batch['filename'][0]
    self.eval_rst[filename] = self.eval_rst.get(filename, 0) + prob_all_pts

    # Save the prediction results in the last epoch
    if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
      full_filename = os.path.join(self.logdir, filename + '.eval.npz')
      curr_folder = os.path.dirname(full_filename)
      if not os.path.exists(curr_folder): os.makedirs(curr_folder)
      np.savez(full_filename, prob=self.eval_rst[filename].numpy())

  def result_callback(self, avg_tracker, epoch):
    ''' Calculate the part mIoU for PartNet and ScanNet.
    '''

    iou_part = 0.0
    avg = avg_tracker.average()

    # Labels smaller than `mask` is ignored. The points with the label 0 in
    # PartNet are background points, i.e., unlabeled points
    mask = self.FLAGS.LOSS.mask + 1
    num_class = self.FLAGS.LOSS.num_class
    for i in range(mask, num_class):
      instc_i = avg['test/intsc_%d' % i]
      union_i = avg['test/union_%d' % i]
      iou_part += instc_i / (union_i + 1.0e-10)
    iou_part = iou_part / (num_class - mask)

    tqdm.write('=> Epoch: %d, test/mIoU_part: %f' % (epoch, iou_part))
    if self.summary_writer:
      self.summary_writer.add_scalar('test/mIoU_part', iou_part, epoch)

  def loss_function(self, logit, label):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logit, label.long())
    return loss

  def accuracy(self, logit, label):
    pred = logit.argmax(dim=1)
    accu = pred.eq(label).float().mean()
    return accu

  def IoU_per_shape(self, logit, label, class_num):
    pred = logit.argmax(dim=1)

    IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
    intsc, union = [None] * class_num, [None] * class_num
    for k in range(class_num):
      pk, lk = pred.eq(k), label.eq(k)
      intsc[k] = torch.sum(torch.logical_and(pk, lk).float())
      union[k] = torch.sum(torch.logical_or(pk, lk).float())

      valid = torch.sum(lk.any()) > 0
      valid_part_num += valid.item()
      IoU += valid * intsc[k] / (union[k] + esp)

    # Calculate the shape IoU for ShapeNet
    IoU /= valid_part_num + esp
    return IoU, intsc, union

  @classmethod
  def update_configs(cls):
    FLAGS = get_config()
    FLAGS.LOSS.mask = -1           # mask the invalid labels


if __name__ == "__main__":
  SegSolver.main()
