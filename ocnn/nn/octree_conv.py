import math
import torch
import torch.nn
from torch.autograd import Function
from typing import List

from ocnn.octree import Octree
from ocnn.utils import scatter_add
from .octree2col import octree2col, col2octree
from .octree_pad import octree_pad, octree_depad


class OctreeConvBase:

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False, max_buffer: int = int(2e8)):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = self.resize_with_last_val(kernel_size)
    self.kernel = self.list2str(self.kernel_size)
    self.stride = stride
    self.nempty = nempty
    self.max_buffer = max_buffer  # about 200M

    self.kdim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
    self.in_conv = in_channels if self.is_conv_layer() else out_channels
    self.out_conv = out_channels if self.is_conv_layer() else in_channels
    self.weights_shape = (self.kdim, self.in_conv, self.out_conv)

  def resize_with_last_val(self, list_in, num=3):
    r''' Resizes the number of elements of :attr:`list_in` to :attr:`num` with
    the last element of :attr:`list_in` if its number of elements is smaller 
    than :attr:`num`.
    '''

    assert (type(list_in) is list and len(list_in) < num + 1)
    for i in range(len(list_in), num):
      list_in.append(list_in[-1])
    return list_in

  def list2str(self, list_in):
    r''' Returns a string representation of :attr:`list_in`
    '''

    out = [str(x) for x in list_in]
    return ''.join(out)

  def is_conv_layer(self):
    r''' Returns :obj:`True` to indicate this is a convolution layer.
    '''

    raise NotImplementedError

  def setup(self, octree: Octree, depth: int):
    r''' Setup the shapes of each tensor. 
    This function MUST be called before :obj:`forward_gemm`, :obj:`backward_gemm`
    and :obj:`weight_gemm`.
    '''

    # The depth of tensors:
    # The in_depth and out_depth are the octree depth of the input and output
    # data; neigh_depth is the octree depth of the neighborhood information, as
    # well as `col` data, neigh_depth is always the same as the depth of larger
    # data when doing octree2col or col2octree.
    self.in_depth = depth
    self.out_depth = depth
    self.neigh_depth = depth
    if self.stride == 2:
      if self.is_conv_layer():
        self.out_depth = depth - 1
      else:
        self.out_depth = depth + 1
        self.neigh_depth = depth + 1

    # The height of tensors
    if self.nempty:
      self.in_h = octree.nnum_nempty[self.in_depth]
      self.out_h = octree.nnum_nempty[self.out_depth]
    else:
      self.in_h = octree.nnum[self.in_depth]
      self.out_h = octree.nnum[self.out_depth]
      if self.stride == 2:
        if self.is_conv_layer():
          self.out_h = octree.nnum_nempty[self.out_depth]
        else:
          self.in_h = octree.nnum_nempty[self.in_depth]
    self.in_shape = (self.in_h, self.in_channels)
    self.out_shape = (self.out_h, self.out_channels)

    # The neighborhood indices
    self.neigh = octree.get_neigh(
        self.neigh_depth, self.kernel, self.stride, self.nempty)

    # The heigh and number of the temporary buffer
    self.buffer_n = 1
    self.buffer_h = self.neigh.shape[0]
    ideal_size = self.buffer_h * self.neigh.shape[1] * self.in_conv
    if ideal_size > self.max_buffer:
      self.buffer_n = (ideal_size + self.max_buffer - 1) // self.max_buffer
      self.buffer_h = (self.buffer_h + self.buffer_n - 1) // self.buffer_n
    self.buffer_shape = (self.buffer_h, self.kdim, self.in_conv)

  def check_and_init(self, data: torch.Tensor):
    r''' Checks the input data and initializes the shape of output data.
    '''

    # Check the shape of input data
    check = tuple(data.shape) == self.in_shape
    torch._assert(check, 'The shape of input data is wrong.')

    # Init the output data
    out = data.new_zeros(self.out_shape)
    return out

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
      out[start:end] = torch.mm(buffer.flatten(1, 2), weights.flatten(0, 1))

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
      buffer = torch.mm(grad[start:end], weights.flatten(0, 1).t())
      buffer = buffer.view(-1, self.buffer_shape[1], self.buffer_shape[2])

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
      out.addmm_(buffer.flatten(1, 2).t(), grad[start:end])

    return out.view(out_shape)


class _OctreeConv(OctreeConvBase):
  r''' Instantiates _OctreeConvBase by overriding `is_conv_layer`
  '''

  def is_conv_layer(self): return True


class _OctreeDeconv(OctreeConvBase):
  r''' Instantiates _OctreeConvBase by overriding `is_conv_layer`
  '''

  def is_conv_layer(self): return False


class OctreeConvFunction(Function):
  r''' Wrap the octree convolution for auto-diff.
  '''

  @staticmethod
  def forward(ctx, data: torch.Tensor, weights: torch.Tensor, octree: Octree,
              depth: int, in_channels: int, out_channels: int,
              kernel_size: List[int] = [3, 3, 3], stride: int = 1,
              nempty: bool = False, max_buffer: int = int(2e8)):
    octree_conv = _OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty, max_buffer)
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

    return (grad_out, grad_w) + (None,) * 8


class OctreeDeconvFunction(Function):
  r''' Wrap the octree deconvolution for auto-diff.
  '''

  @staticmethod
  def forward(ctx, data: torch.Tensor, weights: torch.Tensor, octree: Octree,
              depth: int, in_channels: int, out_channels: int,
              kernel_size: List[int] = [3, 3, 3], stride: int = 1,
              nempty: bool = False, max_buffer: int = int(2e8)):
    octree_deconv = _OctreeDeconv(
        in_channels, out_channels, kernel_size, stride, nempty, max_buffer)
    octree_deconv.setup(octree, depth)
    out = octree_deconv.check_and_init(data)
    out = octree_deconv.backward_gemm(out, data, weights)

    ctx.save_for_backward(data, weights)
    ctx.octree_deconv = octree_deconv
    return out

  @staticmethod
  def backward(ctx, grad):
    data, weights = ctx.saved_tensors
    octree_deconv = ctx.octree_deconv

    grad_out = None
    if ctx.needs_input_grad[0]:
      grad_out = torch.zeros_like(data)
      grad_out = octree_deconv.forward_gemm(grad_out, grad, weights)

    grad_w = None
    if ctx.needs_input_grad[1]:
      grad_w = torch.zeros_like(weights)
      grad_w = octree_deconv.weight_gemm(grad_w, grad, data)

    return (grad_out, grad_w) + (None,) * 8


# alias
octree_conv = OctreeConvFunction.apply
octree_deconv = OctreeDeconvFunction.apply


class OctreeConv(OctreeConvBase, torch.nn.Module):
  r''' Performs octree convolution.

  Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    kernel_size (List(int)): The kernel shape, choose from :obj:`[3]`,
        :obj:`[2]`,`[3,3,3]`, :obj:`[3,1,1]`, :obj:`[1,3,1]`, :obj:`[1,1,3]`,
        :obj:`[2,2,2]`, :obj:`[3,3,1]`, :obj:`[1,3,3]`, and :obj:`[3,1,3]`.
    stride (int): The stride of the convolution (:obj:`1` or :obj:`2`).
    nempty (bool): If True, only performs the convolution on non-empty
        octree nodes.
    direct_method (bool): If True, directly performs the convolution via using
        gemm and octree2col/col2octree. The octree2col/col2octree needs to 
        construct a large matrix, which may consume a lot of memory. If False,
        performs the convolution in a sub-matrix manner, which can save the 
        requied runtime memory.
    use_bias (bool): If True, add a bias term to the convolution.
    max_buffer (int): The maximum number of elements in the buffer, used when 
        :attr:`direct_method` is False.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False, direct_method: bool = False,
               use_bias: bool = False, max_buffer: int = int(2e8)):
    super().__init__(
        in_channels, out_channels, kernel_size, stride, nempty, max_buffer)

    self.direct_method = direct_method
    self.use_bias = use_bias
    self.weights = torch.nn.Parameter(torch.Tensor(*self.weights_shape))
    if self.use_bias:
      self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
    self.reset_parameters()

  def reset_parameters(self):
    r''' Initialize convolution weights with the same method as
    :obj:`torch.nn.init.xavier_uniform_`.

    :obj:`torch.nn.init.xavier_uniform_` initialize a tensor with shape
    :obj:`(out_c, in_c, kdim)`. It can not be used in :class:`OctreeConv` since
    the the shape of :attr:`OctreeConv.weights` is :obj:`(kdim, in_c, out_c)`
    '''

    shape = self.weights.shape  # (kernel_dim, in_conv, out_conv)
    fan_in = shape[0] * shape[1]
    fan_out = shape[0] * shape[2]
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    torch.nn.init.uniform_(self.weights, -a, a)
    if self.use_bias:
      torch.nn.init.zeros_(self.bias)

  def is_conv_layer(self): return True

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''' Defines the octree convolution.

    Args:
      data (torch.Tensor): The input data.
      octree (Octree): The corresponding octree.
      depth (int): The depth of current octree.
    '''

    if self.direct_method:
      col = octree2col(
          data, octree, depth, self.kernel, self.stride, self.nempty)
      out = torch.mm(col.flatten(1), self.weights.flatten(0, 1))
    else:
      out = octree_conv(
          data, self.weights, octree, depth, self.in_channels,
          self.out_channels, self.kernel_size, self.stride, self.nempty,
          self.max_buffer)

    if self.use_bias:
      out += self.bias

    if self.stride == 2 and not self.nempty:
      out = octree_pad(out, octree, depth-1)
    return out

  def extra_repr(self) -> str:
    r''' Sets the extra representation of the module.
    '''

    return ('in_channels={}, out_channels={}, kernel_size={}, stride={}, '
            'nempty={}, bias={}').format(self.in_channels, self.out_channels,
             self.kernel_size, self.stride, self.nempty, self.use_bias)  # noqa


class OctreeDeconv(OctreeConv):
  r''' Performs octree deconvolution.

  Please refer to :class:`OctreeConv` for the meaning of the arguments.
  '''

  def is_conv_layer(self): return False

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''' Defines the octree deconvolution.

    Please refer to :meth:`OctreeConv.forward` for the meaning of the arguments.
    '''

    depth_col = depth
    if self.stride == 2:
      depth_col = depth + 1
      if not self.nempty:
        data = octree_depad(data, octree, depth)

    if self.direct_method:
      col = torch.mm(data, self.weights.flatten(0, 1).t())
      col = col.view(col.shape[0], self.kdim, -1)
      out = col2octree(
          col, octree, depth_col, self.kernel, self.stride, self.nempty)
    else:
      out = octree_deconv(
          data, self.weights, octree, depth, self.in_channels,
          self.out_channels, self.kernel_size, self.stride, self.nempty,
          self.max_buffer)

    if self.use_bias:
      out += self.bias
    return out
