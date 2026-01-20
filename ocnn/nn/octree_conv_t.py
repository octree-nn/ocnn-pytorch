# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from .implicit_gemm import (
  flex_gemm_backward_weight_implicit,
  flex_gemm_forward_implicit,
)
from typing import List, Optional
from ocnn.utils import resize_with_last_val, list2str
from ocnn.octree import Octree
from .octree_pad import octree_pad, octree_depad


class _flexible_gemm(torch.autograd.Function):
  @staticmethod
  def setup_context(ctx, inputs, output):
    data, weight, neighbour, inv_neighbour = inputs
    ctx.save_for_backward(data, weight, neighbour, inv_neighbour)

  @staticmethod
  def forward(data, weight, neighbour, inv_neighbour):
    return flex_gemm_forward_implicit(data, weight, None, neighbour)

  @staticmethod
  def backward(ctx, upstream_grad):
    data, weight, neighbour, inv_neighbour = ctx.saved_tensors
    grad_data, grad_weight = None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
      upstream_grad = upstream_grad.contiguous()
    if ctx.needs_input_grad[0]:
      grad_data = flex_gemm_forward_implicit(
        upstream_grad, weight.permute(2, 1, 0).contiguous(), None, inv_neighbour
      )
    if ctx.needs_input_grad[1]:
      grad_weight = flex_gemm_backward_weight_implicit(
        upstream_grad, data, neighbour
      )
    return grad_data, grad_weight, None, None


def flexible_gemm(
  data: torch.Tensor,
  weight: torch.Tensor,
  neighbour: torch.Tensor,
  inv_neighbour: torch.Tensor,
):
  return _flexible_gemm.apply(data, weight, neighbour, inv_neighbour)


class OctreeConvTriton(torch.nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: List[int] = [3],
    stride: int = 1,
    nempty: bool = False,
    use_bias: bool = False,
  ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = resize_with_last_val(kernel_size)
    self.kernel = list2str(self.kernel_size)
    self.stride = stride
    self.nempty = nempty
    self.use_bias = use_bias

    self.kdim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
    self.weights_shape = (self.out_channels, self.kdim, self.in_channels)

    self.weights = torch.nn.Parameter(torch.empty(*self.weights_shape))
    self.bias = (
      torch.nn.Parameter(torch.empty(self.out_channels))
      if self.use_bias
      else None
    )

  def forward(
    self, data: torch.Tensor, octree: Octree, depth: int
  ) -> torch.Tensor:
    out = flexible_gemm(
      data,
      self.weights,
      octree.get_neigh(depth, self.kernel, self.stride, self.nempty),
      octree.get_inv_neigh(depth, self.kernel, self.stride, self.nempty),
    )
    if self.use_bias:
      out = out + self.bias

    if self.stride == 2 and not self.nempty:
      out = octree_pad(out, octree, depth - 1)

    return out

  def extra_repr(self) -> str:
    r"""Sets the extra representation of the module."""

    return (
      "in_channels={}, out_channels={}, kernel_size={}, stride={}, "
      "nempty={}, bias={}"
    ).format(
      self.in_channels,
      self.out_channels,
      self.kernel_size,
      self.stride,
      self.nempty,
      self.use_bias,
    )


class OctreeDeConvTriton(torch.nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: List[int] = [3],
    stride: int = 1,
    nempty: bool = False,
    use_bias: bool = False,
  ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = resize_with_last_val(kernel_size)
    self.kernel = list2str(self.kernel_size)
    self.stride = stride
    self.nempty = nempty
    self.use_bias = use_bias

    self.kdim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
    self.weights_shape = (self.out_channels, self.kdim, self.in_channels)

    self.weights = torch.nn.Parameter(torch.empty(*self.weights_shape))
    self.bias = (
      torch.nn.Parameter(torch.empty(self.out_channels))
      if self.use_bias
      else None
    )

  def forward(
    self, data: torch.Tensor, octree: Octree, depth: int
  ) -> torch.Tensor:
    if self.stride == 2 and not self.nempty:
      data = octree_depad(data, octree, depth)

    out = flexible_gemm(
      data,
      self.weights,
      octree.get_inv_neigh(depth, self.kernel, self.stride, self.nempty),
      octree.get_neigh(depth, self.kernel, self.stride, self.nempty),
    )
    if self.use_bias:
      out = out + self.bias

    return out

  def extra_repr(self) -> str:
    r"""Sets the extra representation of the module."""

    return (
      "in_channels={}, out_channels={}, kernel_size={}, stride={}, "
      "nempty={}, bias={}"
    ).format(
      self.in_channels,
      self.out_channels,
      self.kernel_size,
      self.stride,
      self.nempty,
      self.use_bias,
    )
