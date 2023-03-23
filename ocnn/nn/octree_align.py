# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch

from ocnn.octree import Octree


def search_value(value: torch.Tensor, key: torch.Tensor, query: torch.Tensor):
  r''' Searches values according to sorted shuffled keys.

  Args:
    value (torch.Tensor): The input tensor with shape (N, C).
    key (torch.Tensor): The key tensor corresponds to :attr:`value` with shape 
        (N,), which contains sorted shuffled keys of an octree.
    query (torch.Tensor): The query tensor, which also contains shuffled keys.
  '''

  # deal with out-of-bound queries, the indices of these queries
  # returned by torch.searchsorted equal to `key.shape[0]`
  out_of_bound = query > key[-1]

  # search
  idx = torch.searchsorted(key, query)
  idx[out_of_bound] = -1   # to avoid overflow when executing the following line
  found = key[idx] == query

  # assign the found value to the output
  out = torch.zeros(query.shape[0], value.shape[1], device=value.device)
  out[found] = value[idx[found]]
  return out


def octree_align(value: torch.Tensor, octree: Octree, octree_query: Octree,
                 depth: int, nempty: bool = False):
  r''' Wraps :func:`octree_align` to take octrees as input for convenience.
  '''

  key = octree.key(depth, nempty)
  query = octree_query.key(depth, nempty)
  assert key.shape[0] == value.shape[0]
  return search_value(value, key, query)
