import os
import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='segnet_d5')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--model', type=str, default='segnet')
parser.add_argument('--mode', type=str, default='randinit')
parser.add_argument('--ckpt', type=str, default='\'\'')
parser.add_argument('--ratios', type=float, default=[1.0], nargs='*')

args = parser.parse_args()
alias = args.alias
gpu = args.gpu
mode = args.mode
ratios = args.ratios
# ratios = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]

module = 'segmentation.py'
script = 'python %s --config configs/seg_shapenet.yaml' % module
data = 'data/shapenet_segmentation'
logdir = 'logs/seg_shapenet'

categories = ['02691156', '02773838', '02954340', '02958343',
              '03001627', '03261776', '03467517', '03624134',
              '03636649', '03642806', '03790512', '03797390',
              '03948459', '04099429', '04225987', '04379243']
names = ['Aero', 'Bag', 'Cap', 'Car',
         'Chair', 'EarPhone', 'Guitar', 'Knife',
         'Lamp', 'Laptop', 'Motor', 'Mug',
         'Pistol', 'Rocket', 'Skate', 'Table']
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
train_num = [2348, 62, 44, 717, 3052, 55, 626, 312,
             1261, 367, 151, 146, 234, 54, 121, 4421]
test_num = [341, 14, 11, 153, 693, 14, 159, 80,
            285, 78, 51, 38, 41, 12, 31, 842]
max_epoches = [300, 1800, 2400, 600, 300, 2000, 600, 600,
               300, 600, 1200, 1200, 600, 1800, 1200, 200]
max_iters = [20000, 3000, 3000, 10000, 20000, 3000, 10000, 5000,
             10000, 5000, 5000, 5000, 5000, 3000, 5000, 20000]


for i in range(len(ratios)):
  for k in range(len(categories)):
    ratio, cat = ratios[i], categories[k]
    mul = 2 if ratios[i] < 0.1 else 1  # longer iterations when data < 10%
    max_epoch = int(max_epoches[k] * ratio * mul)
    step_size1, step_size2 = int(0.5 * max_epoch), int(0.25 * max_epoch)
    test_every_epoch = int(math.ceil(max_epoch * 0.02))
    take = int(math.ceil(train_num[k] * ratio))
    logs = os.path.join(
        logdir, '{}/{}_{}/ratio_{:.2f}'.format(alias, cat, names[k], ratio))

    cmds = [
        script,
        'SOLVER.gpu {},'.format(gpu),
        'SOLVER.logdir {}'.format(logs),
        'SOLVER.max_epoch {}'.format(max_epoch),
        'SOLVER.step_size {},{}'.format(step_size1, step_size2),
        'SOLVER.test_every_epoch {}'.format(test_every_epoch),
        'SOLVER.ckpt {}'.format(args.ckpt),
        'DATA.train.depth {}'.format(args.depth),
        'DATA.train.filelist {}/filelist/{}_train_val.txt'.format(data, cat),
        'DATA.train.take {}'.format(take),
        'DATA.test.depth {}'.format(args.depth),
        'DATA.test.filelist {}/filelist/{}_test.txt'.format(data, cat),
        'MODEL.stages {}'.format(args.depth - 2),
        'MODEL.nout {}'.format(seg_num[k]),
        'MODEL.name {}'.format(args.model),
        'LOSS.num_class {}'.format(seg_num[k])
    ]

    cmd = ' '.join(cmds)
    print('\n', cmd, '\n')
    os.system(cmd)

summary = []
summary.append('names, ' + ', '.join(names) + ', C.mIoU, I.mIoU')
summary.append('train_num, ' + ', '.join([str(x) for x in train_num]))
summary.append('test_num, ' + ', '.join([str(x) for x in test_num]))

for i in range(len(ratios)-1, -1, -1):
  ious = [None] * len(categories)
  for j in range(len(categories)):
    filename = '{}/{}/{}_{}/ratio_{:.2f}/log.csv'.format(
        logdir, alias, categories[j], names[j], ratios[i])
    with open(filename, newline='') as fid:
      lines = fid.readlines()
    last_line = lines[-1]

    pos = last_line.find('test/mIoU:')
    ious[j] = float(last_line[pos+11:pos+16])
  CmIoU = np.array(ious).mean()
  ImIoU = np.sum(np.array(ious)*np.array(test_num)) / np.sum(np.array(test_num))

  ious = [str(iou) for iou in ious] + \
         ['{:.3f}'.format(CmIoU), '{:.3f}'.format(ImIoU)]
  summary.append('Ratio:{:.2f}, '.format(ratios[i]) + ', '.join(ious))

with open('{}/{}/summaries.csv'.format(logdir, alias), 'w') as fid:
  summ = '\n'.join(summary)
  fid.write(summ)
  print(summ)
