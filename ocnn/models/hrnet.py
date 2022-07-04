import torch
from typing import List

import ocnn
from ocnn.octree import Octree


class Branches(torch.nn.Module):

  def __init__(self, channels: List[int], resblk_num: int, nempty: bool = False):
    super().__init__()
    self.channels = channels
    self.resblk_num = resblk_num
    bottlenecks = [4 if c < 256 else 8 for c in channels]  # to save parameters
    self.resblocks = torch.nn.ModuleList([
        ocnn.modules.OctreeResBlocks(ch, ch, resblk_num, bnk, nempty=nempty)
        for ch, bnk in zip(channels, bottlenecks)])

  def forward(self, datas: List[torch.Tensor], octree: Octree, depth: int):
    num = len(self.channels)
    torch._assert(len(datas) == num, 'Error')

    out = [None] * num
    for i in range(num):
      depth_i = depth - i
      out[i] = self.resblocks[i](datas[i], octree, depth_i)
    return out


class TransFunc(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int, nempty: bool = False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.nempty = nempty
    if in_channels != out_channels:
      self.conv1x1 = ocnn.modules.Conv1x1BnRelu(in_channels, out_channels)

  def forward(self, data: torch.Tensor, octree: Octree,
              in_depth: int, out_depth: int):
    out = data
    if in_depth > out_depth:
      for d in range(in_depth, out_depth, -1):
        out = ocnn.nn.octree_max_pool(out, octree, d, self.nempty)
      if self.in_channels != self.out_channels:
        out = self.conv1x1(out)

    if in_depth < out_depth:
      if self.in_channels != self.out_channels:
        out = self.conv1x1(out)
      for d in range(in_depth, out_depth, 1):
        out = ocnn.nn.octree_upsample(out, octree, d, self.nempty)
    return out


class Transitions(torch.nn.Module):

  def __init__(self, channels: List[int], nempty: bool = False):
    super().__init__()
    self.channels = channels
    self.nempty = nempty

    num = len(self.channels)
    self.trans_func = torch.nn.ModuleList()
    for i in range(num - 1):
      for j in range(num):
        self.trans_func.append(TransFunc(channels[i], channels[j], nempty))

  def forward(self, data: List[torch.Tensor], octree: Octree, depth: int):
    num = len(self.channels)
    features = [[None] * (num - 1) for _ in range(num)]
    for i in range(num - 1):
      for j in range(num):
        k = i * num + j
        in_depth = depth - i
        out_depth = depth - j
        features[j][i] = self.trans_func[k](
            data[i], octree, in_depth, out_depth)

    out = [None] * num
    for j in range(num):
      # In the original tensorflow implmentation, the relu is added here,
      # instead of Line 77
      out[j] = torch.stack(features[j], dim=0).sum(dim=0)
    return out


class FrontLayer(torch.nn.Module):

  def __init__(self, channels: List[int], nempty: bool = False):
    super().__init__()
    self.channels = channels
    self.num = len(channels) - 1
    self.nempty = nempty

    self.conv = torch.nn.ModuleList([
        ocnn.modules.OctreeConvBnRelu(channels[i], channels[i+1], nempty=nempty)
        for i in range(self.num)])
    self.maxpool = torch.nn.ModuleList([
        ocnn.nn.OctreeMaxPool(nempty) for i in range(self.num - 1)])

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    out = data
    for i in range(self.num - 1):
      depth_i = depth - i
      out = self.conv[i](out, octree, depth_i)
      out = self.maxpool[i](out, octree, depth_i)
    out = self.conv[-1](out, octree, depth - self.num + 1)
    return out


class ClsHeader(torch.nn.Module):

  def __init__(self, channels: List[int], out_channels: int, nempty: bool = False):
    super().__init__()
    self.channels = channels
    self.out_channels = out_channels
    self.nempty = nempty

    in_channels = int(torch.Tensor(channels).sum())
    self.conv1x1 = ocnn.modules.Conv1x1BnRelu(in_channels, 1024)
    self.global_pool = ocnn.nn.OctreeGlobalPool(nempty)
    self.header = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(1024, out_channels, bias=True))
    # self.header = torch.nn.Sequential(
    #     ocnn.modules.FcBnRelu(512, 256),
    #     torch.nn.Dropout(p=0.5),
    #     torch.nn.Linear(256, out_channels))

  def forward(self, data: List[torch.Tensor], octree: Octree, depth: int):
    full_depth = 2
    num = len(data)
    for i in range(num):
      depth_i = depth - i
      for d in range(depth_i, full_depth, -1):
        data[i] = ocnn.nn.octree_max_pool(data[i], octree, d, self.nempty)

    out = torch.cat(data, dim=1)
    out = self.conv1x1(out)
    out = self.global_pool(out, octree, full_depth)
    logit = self.header(out)
    return logit


class HRNet(torch.nn.Module):
  r''' Octree-based HRNet for classification and segmentation. '''

  def __init__(self, in_channels: int, out_channels: int, stages: int = 3,
               interp: str = 'linear', nempty: bool = False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.interp = interp
    self.nempty = nempty
    self.stages = stages

    self.resblk_num = 3
    self.channels = [128, 256, 512, 512]

    self.front = FrontLayer([in_channels, 32, self.channels[0]], nempty)
    self.branches = torch.nn.ModuleList([
        Branches(self.channels[:i+1], self.resblk_num, nempty)
        for i in range(stages)])
    self.transitions = torch.nn.ModuleList([
        Transitions(self.channels[:i+2], nempty)
        for i in range(stages-1)])

    self.cls_header = ClsHeader(self.channels[:stages], out_channels, nempty)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''
    convs = [self.front(data, octree, depth)]
    depth = depth - 1  # the data is downsampled in `front`
    for i in range(self.stages):
      convs = self.branches[i](convs, octree, depth)
      if i < self.stages - 1:
        convs = self.transitions[i](convs, octree, depth)

    logits = self.cls_header(convs, octree, depth)

    return logits
