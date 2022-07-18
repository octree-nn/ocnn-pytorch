import torch
import torch.utils.checkpoint
from typing import List

import ocnn
from ocnn.nn import OctreeConv, OctreeDeconv
from ocnn.octree import Octree


bn_momentum, bn_eps = 0.01, 0.001    # the default value of Tensorflow 1.x
# bn_momentum, bn_eps = 0.1, 1e-05   # the default value of pytorch


def ckpt_conv_wrapper(conv_op, data, octree):
  # The dummy tensor is a workaround when the checkpoint is used for the first conv layer:
  # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
  dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

  def conv_wrapper(data, octree, dummy_tensor):
    return conv_op(data, octree)

  return torch.utils.checkpoint.checkpoint(conv_wrapper, data, octree, dummy)


class OctreeConvBn(torch.nn.Module):
  r''' A sequence of :class:`OctreeConv` and :obj:`BatchNorm`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.bn(out)
    return out


class OctreeConvBnRelu(torch.nn.Module):
  r''' A sequence of :class:`OctreeConv`, :obj:`BatchNorm`, and :obj:`Relu`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.bn(out)
    out = self.relu(out)
    return out


class OctreeDeconvBnRelu(torch.nn.Module):
  r''' A sequence of :class:`OctreeDeconv`, :obj:`BatchNorm`, and :obj:`Relu`.

  Please refer to :class:`ocnn.nn.OctreeDeconv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.deconv = OctreeDeconv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.deconv(data, octree, depth)
    out = self.bn(out)
    out = self.relu(out)
    return out


class Conv1x1(torch.nn.Module):
  r''' Performs a convolution with kernel :obj:`(1,1,1)`.

  The shape of octree features is :obj:`(N, C)`, where :obj:`N` is the node
  number and :obj:`C` is the feature channel. Therefore, :class:`Conv1x1` can be
  implemented with :class:`torch.nn.Linear`.
  '''

  def __init__(self, in_channels: int, out_channels: int, use_bias: bool = False):
    super().__init__()
    self.linear = torch.nn.Linear(in_channels, out_channels, use_bias)

  def forward(self, data: torch.Tensor):
    r''''''

    return self.linear(data)


class Conv1x1Bn(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1` and :class:`BatchNorm`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)

  def forward(self, data: torch.Tensor):
    r''''''

    out = self.conv(data)
    out = self.bn(out)
    return out


class Conv1x1BnRelu(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1`, :class:`BatchNorm` and :class:`Relu`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor):
    r''''''

    out = self.conv(data)
    out = self.bn(out)
    out = self.relu(out)
    return out


class FcBnRelu(torch.nn.Module):
  r''' A sequence of :class:`FC`, :class:`BatchNorm` and :class:`Relu`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.flatten = torch.nn.Flatten(start_dim=1)
    self.fc = torch.nn.Linear(in_channels, out_channels, bias=False)
    self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data):
    r''''''

    out = self.flatten(data)
    out = self.fc(out)
    out = self.bn(out)
    out = self.relu(out)
    return out


class InputFeature(torch.nn.Module):
  r''' Returns the initial input feature stored in octree.

  Args:
    feature (str): A string used to indicate which features to extract from the
        input octree. If the character :obj:`N` is in :attr:`feature`, the
        normal signal is extracted (3 channels). Similarly, if :obj:`D` is in
        :attr:`feature`, the local displacement is extracted (1 channels). If
        :obj:`L` is in :attr:`feature`, the local coordinates of the averaged
        points in each octree node is extracted (3 channels). If :attr:`P` is in
        :attr:`feature`, the global coordinates are extracted (3 channels). If
        :attr:`F` is in :attr:`feature`, other features (like colors) are
        extracted (k channels).
    nempty (bool): If false, gets the features of all octree nodes. 
  '''

  def __init__(self, feature: str = 'NDF', nempty: bool = False):
    super().__init__()
    self.nempty = nempty
    self.feature = feature.upper()

  def forward(self, octree: Octree):
    r''''''

    features = list()
    depth = octree.depth
    if 'N' in self.feature:
      features.append(octree.normals[depth])

    if 'L' in self.feature or 'D' in self.feature:
      local_points = octree.points[depth].frac() - 0.5

    if 'D' in self.feature:
      dis = torch.sum(local_points * octree.normals[depth], dim=1, keepdim=True)
      features.append(dis)

    if 'L' in self.feature:
      features.append(local_points)

    if 'P' in self.feature:
      scale = 2 ** (1 - depth)   # normalize [0, 2^depth] -> [-1, 1]
      global_points = octree.points[depth] * scale - 1.0
      features.append(global_points)

    if 'F' in self.feature:
      features.append(octree.features[depth])

    out = torch.cat(features, dim=1)
    if not self.nempty:
      out = ocnn.nn.octree_pad(out, octree, depth)
    return out

  def extra_repr(self) -> str:
    r''''''
    return 'feature={}, nempty={}'.format(self.feature, self.nempty)
