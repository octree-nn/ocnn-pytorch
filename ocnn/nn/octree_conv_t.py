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

import ocnn
from ocnn.octree import Octree
from ocnn.utils import xavier_uniform_, resize_with_last_val, list2str

# Conditionally import Triton kernels, only available on GPU
try:
  from ocnn.nn.kernels import (
      conv_fwd_implicit_gemm_splitk,
      conv_bwd_implicit_gemm_splitk)
except ImportError:
  conv_fwd_implicit_gemm_splitk = None
  conv_bwd_implicit_gemm_splitk = None


class OctreeConvTritonFunction(Function):
  r''' Wrap the octree convolution for auto-diff.
  '''

  @staticmethod
  def forward(ctx, data: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor,
              neigh: torch.Tensor):
    data = data.contiguous()
    weights = weights.contiguous()
    neigh = neigh.contiguous()
    if bias is not None:
      bias = bias.contiguous()

    out = conv_fwd_implicit_gemm_splitk(data, weights, bias, neigh)
    ctx.save_for_backward(data, weights, bias, neigh)
    return out

  @staticmethod
  def backward(ctx, grad):
    data, weights, bias, neigh = ctx.saved_tensors
    grad = grad.contiguous()
    grad_input, grad_weight, grad_bias = conv_bwd_implicit_gemm_splitk(
        grad, data, weights, bias, neigh, ctx.needs_input_grad)
    return grad_input, grad_weight, grad_bias, None


# alias
octree_conv_triton = OctreeConvTritonFunction.apply


class OctreeConvTriton(torch.nn.Module):
  r''' Performs octree convolution.

  Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    kernel_size (List(int)): The kernel shape, only :obj:`[3]` and :obj:`[3,3,3]`
        are supported now for the triton implementation.
    stride (int): The stride of the convolution, only :obj:`1` is supported now.
    nempty (bool): If True, only performs the convolution on non-empty octree
        nodes; otherwise, performs the convolution on all octree nodes.
    use_bias (bool): If True, add a bias term to the convolution.

  .. note::
    Each non-empty octree node has exactly 8 children nodes, among which some
    children nodes are non-empty and some are empty. If :attr:`nempty` is true,
    the convolution is performed on non-empty octree nodes only, which is exactly
    the same as SparseConvNet and MinkowsiNet; if :attr:`nempty` is false, the
    convolution is performed on all octree nodes, which is essential for shape
    reconstruction tasks and can also be used in classification and segmentation
    (with slightly better performance and larger memory cost).
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False, method: str = 'triton',
               use_bias: bool = False, max_buffer: int = int(2e8)):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = resize_with_last_val(kernel_size)
    self.kernel = list2str(self.kernel_size)
    self.stride = stride
    self.nempty = nempty
    self.use_bias = use_bias
    assert self.stride == 1, 'Only stride=1 is supported now.'
    assert self.kernel == '333', 'Only kernel_size=[3,3,3] is supported now.'

    self.kdim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
    self.weights_shape = (self.kdim, self.in_channels, self.out_channels)
    self.weights = torch.nn.Parameter(torch.Tensor(*self.weights_shape))
    self.bias = (torch.nn.Parameter(torch.Tensor(self.out_channels))
                 if use_bias else None)
    self.reset_parameters()

  def reset_parameters(self):
    xavier_uniform_(self.weights)
    if self.use_bias:
      torch.nn.init.zeros_(self.bias)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''' Defines the octree convolution.

    Args:
      data (torch.Tensor): The input data.
      octree (Octree): The corresponding octree.
      depth (int): The depth of current octree.
    '''

    # TODO: remove the permute operation by changing the kernel implementation
    weight = self.weights.permute(2, 0, 1)   # (V,Ci,Co) -> (Co,V,Ci)
    neigh = octree.get_neigh(depth, self.kernel, self.stride, self.nempty)
    out = octree_conv_triton(data, weight, self.bias, neigh)
    return out

  def extra_repr(self) -> str:
    r''' Sets the extra representation of the module.
    '''

    return ('triton, in_channels={}, out_channels={}, kernel_size={}, stride={}, '
            'nempty={}, bias={}').format(self.in_channels, self.out_channels,
             self.kernel_size, self.stride, self.nempty, self.use_bias)  # noqa


# alias
OctreeConvT = OctreeConvTriton


def convert_conv_triton(module: torch.nn.Module) -> torch.nn.Module:
  r''' Convert OctreeConv modules to OctreeConvTriton modules in a network.

  Args:
    module (torch.nn.Module): The input module.
  '''

  module_out = module
  if (isinstance(module, ocnn.nn.OctreeConv) and
          module.stride == 1 and module.kernel_size == [3, 3, 3]):
    module_out = OctreeConvTriton(
        module.in_channels, module.out_channels, module.kernel_size,
        module.stride, module.nempty, use_bias=module.use_bias,)
    with torch.no_grad():
      module_out.weights = module.weights
      if module.use_bias:
        module_out.bias = module.bias

  for name, child in module.named_children():
    module_out.add_module(name, convert_conv_triton(child))
  del module
  return module_out
