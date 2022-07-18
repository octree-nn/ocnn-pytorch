import torch
from typing import Optional

from ocnn.octree import Octree


class OctreeDropPath(torch.nn.Module):
  r'''Drop paths (Stochastic Depth) per sample when applied in main path of 
  residual blocks, following the logic of :func:`timm.models.layers.DropPath`.

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
    r''''''

    if self.drop_prob <= 0.0 or not self.training:
      return data

    batch_size = octree.batch_size
    keep_prob = 1 - self.drop_prob
    rnd_tensor = torch.rand(batch_size, 1, dtype=data.dtype, device=data.device)
    rnd_tensor = torch.floor(rnd_tensor + keep_prob)
    if keep_prob > 0.0 and self.scale_by_keep:
      rnd_tensor.div_(keep_prob)

    if batch_id is None:
      batch_id = octree.batch_id(depth, self.nempty)
    drop_mask = rnd_tensor[batch_id]
    output = data * drop_mask
    return output

  def extra_repr(self) -> str:
    return ('drop_prob={:.4f}, nempty={}, scale_by_keep={}').format(
            self.drop_prob, self.nempty, self.scale_by_keep)  # noqa
