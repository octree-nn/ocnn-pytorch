import torch

from .octree import merge_octrees, merge_points


class CollateBatch:
  r''' Merge a list of octrees and points into a batch.
  '''

  def __init__(self, merge_points: bool = False):
    self.merge_points = merge_points

  def __call__(self, batch: list):
    assert type(batch) == list

    outputs = {}
    for key in batch[0].keys():
      outputs[key] = [b[key] for b in batch]

      # Merge a batch of octrees into one super octree
      if 'octree' in key:
        octree = merge_octrees(outputs[key])
        # NOTE: remember to construct the neighbor indices
        octree.construct_all_neigh()
        outputs[key] = octree

      # Merge a batch of points
      if 'points' in key and self.merge_points:
        outputs[key] = merge_points(outputs[key])

      # Convert the labels to a Tensor
      if 'label' in key:
        outputs['label'] = torch.tensor(outputs[key])

    return outputs
