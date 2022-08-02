# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn
from torch.autograd import Function
from typing import List

from ocnn.octree import Octree
from ocnn.utils import scatter_add, xavier_uniform_
from .octree_pad import octree_pad
from .octree_conv import OctreeConvBase


class OctreeDWConvBase(OctreeConvBase):

  def __init__(self, in_channels: int, kernel_size: List[int] = [3],
               stride: int = 1, nempty: bool = False,
               max_buffer: int = int(2e8)):
    super().__init__(
        in_channels, in_channels, kernel_size, stride, nempty, max_buffer)
    self.weights_shape = (self.kdim, 1, self.out_channels)

  def is_conv_layer(self): return True

  def forward_gemm(self, out: torch.Tensor, data: torch.Tensor,
                   weights: torch.Tensor):
    r''' Peforms the forward pass of octree-based convolution.
    '''

    # Initialize the buffer
    buffer = data.new_empty(self.buffer_shape)

    # Loop over each sub-matrix
    for i in range(self.buffer_n):
      start = i * self.buffer_h
      end = (i + 1) * self.buffer_h

      # The boundary case in the last iteration
      if end > self.neigh.shape[0]:
        dis = end - self.neigh.shape[0]
        end = self.neigh.shape[0]
        buffer, _ = buffer.split([self.buffer_h-dis, dis])

      # Perform octree2col
      neigh_i = self.neigh[start:end]
      valid = neigh_i >= 0
      buffer.fill_(0)
      buffer[valid] = data[neigh_i[valid]]

      # The sub-matrix gemm
      # out[start:end] = torch.mm(buffer.flatten(1, 2), weights.flatten(0, 1))
      out[start:end] = torch.einsum('ikc,kc->ic', buffer, weights.flatten(0, 1))
    return out

  def backward_gemm(self, out: torch.Tensor, grad: torch.Tensor,
                    weights: torch.Tensor):
    r''' Performs the backward pass of octree-based convolution. 
    '''

    # Loop over each sub-matrix
    for i in range(self.buffer_n):
      start = i * self.buffer_h
      end = (i + 1) * self.buffer_h

      # The boundary case in the last iteration
      if end > self.neigh.shape[0]:
        end = self.neigh.shape[0]

      # The sub-matrix gemm
      # buffer = torch.mm(grad[start:end], weights.flatten(0, 1).t())
      # buffer = buffer.view(-1, self.buffer_shape[1], self.buffer_shape[2])
      buffer = torch.einsum(
          'ic,kc->ikc', grad[start:end], weights.flatten(0, 1))

      # Performs col2octree
      neigh_i = self.neigh[start:end]
      valid = neigh_i >= 0
      out = scatter_add(buffer[valid], neigh_i[valid], dim=0, out=out)

    return out

  def weight_gemm(self, out: torch.Tensor, data: torch.Tensor, grad: torch.Tensor):
    r''' Computes the gradient of the weight matrix.
    '''

    # Record the shape of out
    out_shape = out.shape
    out = out.flatten(0, 1)

    # Initialize the buffer
    buffer = data.new_empty(self.buffer_shape)

    # Loop over each sub-matrix
    for i in range(self.buffer_n):
      start = i * self.buffer_h
      end = (i + 1) * self.buffer_h

      # The boundary case in the last iteration
      if end > self.neigh.shape[0]:
        d = end - self.neigh.shape[0]
        end = self.neigh.shape[0]
        buffer, _ = buffer.split([self.buffer_h-d, d])

      # Perform octree2col
      neigh_i = self.neigh[start:end]
      valid = neigh_i >= 0
      buffer.fill_(0)
      buffer[valid] = data[neigh_i[valid]]

      # Accumulate the gradient via gemm
      # out.addmm_(buffer.flatten(1, 2).t(), grad[start:end])
      out += torch.einsum('ikc,ic->kc', buffer, grad[start:end])
    return out.view(out_shape)


class OctreeDWConvFunction(Function):
  r''' Wrap the octree convolution for auto-diff.
  '''

  @staticmethod
  def forward(ctx, data: torch.Tensor, weights: torch.Tensor, octree: Octree,
              depth: int, in_channels: int, kernel_size: List[int] = [3, 3, 3],
              stride: int = 1, nempty: bool = False, max_buffer: int = int(2e8)):
    octree_conv = OctreeDWConvBase(
        in_channels, kernel_size, stride, nempty, max_buffer)
    octree_conv.setup(octree, depth)
    out = octree_conv.check_and_init(data)
    out = octree_conv.forward_gemm(out, data, weights)

    ctx.save_for_backward(data, weights)
    ctx.octree_conv = octree_conv
    return out

  @staticmethod
  def backward(ctx, grad):
    data, weights = ctx.saved_tensors
    octree_conv = ctx.octree_conv

    grad_out = None
    if ctx.needs_input_grad[0]:
      grad_out = torch.zeros_like(data)
      grad_out = octree_conv.backward_gemm(grad_out, grad, weights)

    grad_w = None
    if ctx.needs_input_grad[1]:
      grad_w = torch.zeros_like(weights)
      grad_w = octree_conv.weight_gemm(grad_w, data, grad)

    return (grad_out, grad_w) + (None,) * 7


# alias
octree_dwconv = OctreeDWConvFunction.apply


class OctreeDWConv(OctreeDWConvBase, torch.nn.Module):
  r''' Performs octree-based depth-wise convolution.

  Please refer to :class:`ocnn.nn.OctreeConv` for the meaning of the arguments.
  '''

  def __init__(self, in_channels: int, kernel_size: List[int] = [3],
               stride: int = 1, nempty: bool = False, use_bias: bool = False,
               max_buffer: int = int(2e8)):
    super().__init__(in_channels, kernel_size, stride, nempty, max_buffer)

    self.use_bias = use_bias
    self.weights = torch.nn.Parameter(torch.Tensor(*self.weights_shape))
    if self.use_bias:
      self.bias = torch.nn.Parameter(torch.Tensor(in_channels))
    self.reset_parameters()

  def reset_parameters(self):
    xavier_uniform_(self.weights)
    if self.use_bias:
      torch.nn.init.zeros_(self.bias)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = octree_dwconv(
        data, self.weights, octree, depth, self.in_channels,
        self.kernel_size, self.stride, self.nempty, self.max_buffer)

    if self.use_bias:
      out += self.bias

    if self.stride == 2 and not self.nempty:
      out = octree_pad(out, octree, depth-1)
    return out

  def extra_repr(self) -> str:
    return ('in_channels={}, out_channels={}, kernel_size={}, stride={}, '
            'nempty={}, bias={}').format(self.in_channels, self.out_channels,
             self.kernel_size, self.stride, self.nempty, self.use_bias)  # noqa
