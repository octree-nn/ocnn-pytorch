import torch
import torch.nn.functional as F
import ocnn

from solver import Solver, Dataset
from datasets import get_modelnet40_dataset


class ClsSolver(Solver):

  def get_model(self, flags):
    stages = flags.depth - 2
    if flags.name.lower() == 'lenet':
      model = ocnn.models.LeNet(flags.channel, flags.nout, stages=stages)
    elif flags.name.lower() == 'resnet':
      model = ocnn.models.ResNet(flags.channel, flags.nout, stages=stages,
                                 resblock_num=flags.resblock_num)
    else:
      raise ValueError
    return model

  def get_dataset(self, flags):
    return get_modelnet40_dataset(flags)

  def forward(self, batch):
    octree, label = batch['octree'].cuda(), batch['label'].cuda()
    logits = self.model(octree)
    log_softmax = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(log_softmax, label)
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
