import torch
from typing import Optional


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


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = 'add') -> torch.Tensor:
  r''' Reduces all values from the :attr:`src` tensor into :attr:`out` at the
  indices specified in the :attr:`index` tensor along a given axis :attr:`dim`.
  This is just a wrapper of `torch.Tensor.scatter_()
  <https://pytorch.org/docs/1.10/generated/torch.Tensor.scatter_.html#torch-tensor-scatter>`_
  in a boardcasting fashion.

  Args:
    src (torch.Tensor): The source tensor.
    index (torch.Tensor): The indices of elements to scatter.
    dim (torch.Tensor): The axis along which to index, (default: :obj:`-1`).
    out (torch.Tensor or None): The destination tensor.
    dim_size (int or None): If :attr:`out` is not given, automatically create
        output with size :attr:`dim_size` at dimension :attr:`dim`. If
        :attr:`dim_size` is not given, a minimal sized output tensor according
        to :obj:`index.max() + 1` is returned.
    reduce (str): The reduce operation to apply, choose from :obj:`add` and
        :obj:`mul`, (default: :obj:`add`).
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

  return out.scatter_(dim, index, src, reduce)
