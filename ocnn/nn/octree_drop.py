# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional

from ocnn.octree import Octree


class OctreeDropPath(torch.nn.Module):
  r''' Drop paths (Stochastic Depth) `per octree` when applied in main path
  of residual blocks.

  Args:
    drop_prob (int): The probability of drop paths.
    nempty (bool): Indicate whether the input data only contains features of the
        non-empty octree nodes or not.
    scale_by_keep (bool): Whether to scale the kept features proportionally.
  '''

  def __init__(self, drop_prob: float = 0.0, nempty: bool = False,
               scale_by_keep: bool = True):
    super().__init__()

    self.drop_prob = drop_prob
    self.nempty = nempty
    self.scale_by_keep = scale_by_keep

  def forward(self, data: torch.Tensor, octree: Octree, depth: int,
              batch_id: Optional[torch.Tensor] = None):
    r''' Defines the drop path forward function.

    Args:
      data (torch.Tensor): The input features of shape (N, C), where N is the
          number of octree nodes and C is the number of channels.
      octree (Octree): The input octree.
      depth (int): The depth of the octree layer.
      batch_id (torch.Tensor, optional): The batch indices of the octree nodes.
          If not provided, it will be extracted from the octree.
    '''

    if self.drop_prob <= 0.0 or not self.training:
      return data

    keep_prob = 1 - self.drop_prob
    batch_size = octree.batch_size
    rnd_tensor = data.new_empty(batch_size, 1).bernoulli_(keep_prob)
    if keep_prob > 0.0 and self.scale_by_keep:
      rnd_tensor.div_(keep_prob)

    if batch_id is None:
      batch_id = octree.batch_id(depth, self.nempty)
    drop_mask = rnd_tensor[batch_id].to(data.dtype)
    out = data * drop_mask
    return out

  def extra_repr(self) -> str:
    return ('drop_prob={:.3f}, nempty={}, scale_by_keep={}').format(
            self.drop_prob, self.nempty, self.scale_by_keep)  # noqa


class DropPath(torch.nn.Module):
  r''' Drop paths (Stochastic Depth) `per token`  when applied in main path
  of residual blocks, following the logic of :func:`timm.models.layers.DropPath`.

  Args:
    drop_prob (int): The probability of drop paths.
    scale_by_keep (bool): Whether to scale the kept features proportionally.
  '''

  def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True, **kwargs):
    super().__init__()
    self.drop_prob = drop_prob
    self.scale_by_keep = scale_by_keep

  def forward(self, data: torch.Tensor, **kwargs):
    r''' Defines the drop path forward function.

    Args:
      data (torch.Tensor): The input features of shape (N, C), where N is the
          number of tokens and C is the number of channels.
    '''

    if self.drop_prob <= 0.0 or not self.training:
      return data

    keep_prob = 1 - self.drop_prob
    rnd_tensor = data.new_empty(data.shape[0], 1).bernoulli_(keep_prob)
    if keep_prob > 0.0 and self.scale_by_keep:
      rnd_tensor.div_(keep_prob)
    out = data * rnd_tensor.to(data.dtype)
    return out

  def extra_repr(self) -> str:
    return ('drop_prob={:.3f}, scale_by_keep={}').format(
             self.drop_prob, self.scale_by_keep)  # noqa
