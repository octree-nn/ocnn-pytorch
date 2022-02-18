import math
import torch
import torch.nn
from typing import List

from .octree2col import octree2col, col2octree
from .octree_pad import octree_pad, octree_depad


def resize_with_last_val(list_in, num=3):
  r''' Resizes the number of elements of :attr:`list_in` to :attr:`num` with the
  last element of :attr:`list_in` if its number of elements is smaller than
  :attr:`num`.
  '''

  assert (type(list_in) is list and len(list_in) < num + 1)
  for i in range(len(list_in), num):
    list_in.append(list_in[-1])
  return list_in


def list2str(list_in):
  r''' Returns a string representation of :attr:`list_in`
  '''

  out = [str(x) for x in list_in]
  return ''.join(out)


class OctreeConv(torch.nn.Module):
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

  .. note::
    There is no bias term in the convolution for simplicity.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = resize_with_last_val(kernel_size)
    self.kernel = list2str(self.kernel_size)
    self.stride = stride
    self.nempty = nempty

    self.kdim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
    in_c = in_channels if self.is_conv_layer() else out_channels
    out_c = out_channels if self.is_conv_layer() else in_channels
    self.weights = torch.nn.Parameter(torch.Tensor(self.kdim, in_c, out_c))
    self.reset_parameters()

  def reset_parameters(self):
    r''' Initialize convolution weights with the same method as
    :obj:`torch.nn.init.xavier_uniform_`.

    :obj:`torch.nn.init.xavier_uniform_` initialize a tensor with shape
    :obj:`(out_c, in_c, kdim)`. It can not be used in :class:`OctreeConv` since
    the the shape of :attr:`OctreeConv.weights` is :obj:`(kdim, in_c, out_c)`
    '''

    shape = self.weights.shape
    fan_in = shape[0] * shape[1]
    fan_out = shape[0] * shape[2]
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
      return self.weights.uniform_(-a, a)

  def is_conv_layer():
    r''' Returns :obj:`True` to indicate this is a convolution layer.
    '''

    return True

  def forward(self, data, octree, depth):
    r''' Defines the octree convolution.

    Args:
      data (torch.Tensor): The input data.
      octree (Octree): The corresponding octree.
      depth (int): The depth of current octree.
    '''

    col = octree2col(data, octree, depth, self.kernel, self.stride, self.nempty)
    out = torch.mm(col.flatten(1), self.weights.flatten(0, 1))

    if self.stride == 2 and not self.nempty:
      out = octree_pad(out, octree, depth-1)
    return out

  def extra_repr(self) -> str:
    r''' Sets the extra representation of the module.
    '''

    return ('in_channels={}, out_channels={}, kernel_size={}, stride={}, '
            'nempty={}').format(self.in_channels, self.out_channels,
             self.kernel_size, self.stride, self.nempty)  # noqa


class OctreeDeconv(OctreeConv):
  r''' Performs octree deconvolution.

  Please refer to :class:`OctreeConv` for the meaning of the arguments.
  '''

  def is_conv_layer(self):
    r''' Returns :obj:`False` to indicate this is a deconvolution layer.
    '''

    return False

  def forward(self, data, octree, depth):
    r''' Defines the octree deconvolution.

    Please refer to :meth:`OctreeConv.forward` for the meaning of the arguments.
    '''

    if self.stride == 2 and not self.nempty:
      data = octree_depad(data, octree, depth)
      depth = depth + 1

    col = torch.mm(data, self.weights.flatten(0, 1).t())
    col = col.view(col.shape[0], self.kdim, -1)
    out = col2octree(col, octree, depth, self.kernel, self.stride, self.nempty)
    return out
