import torch
from typing import Optional

__all__ = ['meshgrid', 'cumsum', 'scatter_add', ]
classes = __all__


def meshgrid(*tensors, indexing: Optional[str] = None):
  r''' Wraps :func:`torch.meshgrid` for compatibility.
  '''

  version = torch.__version__.split('.')
  larger_than_190 = int(version[0]) > 0 and int(version[1]) > 9

  if larger_than_190:
    return torch.meshgrid(*tensors, indexing=indexing)
  else:
    return torch.meshgrid(*tensors)


def cumsum(data: torch.Tensor, dim: int, exclusive: bool = False):
  r''' Extends :func:`torch.cumsum` with the input argument :attr:`exclusive`.

  Args:
    data (torch.Tensor): The input data.
    dim (int): The dimension to do the operation over.
    exclusive (bool): If false, the behavior is the same as :func:`torch.cumsum`;
        if true, returns the cumulative sum exclusively. Note that if ture,
        the shape of output tensor is larger by 1 than :attr:`data` in the
        dimension where the computation occurs.
  '''

  out = torch.cumsum(data, dim)

  if exclusive:
    size = list(data.size())
    size[dim] = 1
    zeros = data.new_zeros(size)
    out = torch.cat([zeros, out], dim)
  return out


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
  r''' Broadcast :attr:`src` according to :attr:`other`, originally from the 
  library `pytorch_scatter`.
  '''

  if dim < 0:
    dim = other.dim() + dim

  if src.dim() == 1:
    for _ in range(0, dim):
      src = src.unsqueeze(0)
  for _ in range(src.dim(), other.dim()):
    src = src.unsqueeze(-1)

  src = src.expand_as(other)
  return src


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,) -> torch.Tensor:
  r''' Reduces all values from the :attr:`src` tensor into :attr:`out` at the
  indices specified in the :attr:`index` tensor along a given axis :attr:`dim`.
  This is just a wrapper of :func:`torch.scatter` in a boardcasting fashion.

  Args:
    src (torch.Tensor): The source tensor.
    index (torch.Tensor): The indices of elements to scatter.
    dim (torch.Tensor): The axis along which to index, (default: :obj:`-1`).
    out (torch.Tensor or None): The destination tensor.
    dim_size (int or None): If :attr:`out` is not given, automatically create
        output with size :attr:`dim_size` at dimension :attr:`dim`. If
        :attr:`dim_size` is not given, a minimal sized output tensor according
        to :obj:`index.max() + 1` is returned.
    '''

  index = broadcast(index, src, dim)

  if out is None:
    size = list(src.size())
    if dim_size is not None:
      size[dim] = dim_size
    elif index.numel() == 0:
      size[dim] = 0
    else:
      size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)

  return out.scatter_add_(dim, index, src)
