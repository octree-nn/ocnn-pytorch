import torch
from typing import Optional

__all__ = ['torch_meshgrid']
classes = __all__


def torch_meshgrid(*tensors, indexing: Optional[str] = None):
  r''' Wraps :func:`torch.meshgrid` for compatibility.
  '''

  version = [int(x) for x in torch.__version__.split('.')]
  larger_than_190 = version[0] > 0 and version[1] > 9

  if larger_than_190:
    return torch.meshgrid(*tensors, indexing=indexing)
  else:
    return torch.meshgrid(*tensors)
