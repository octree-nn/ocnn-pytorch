# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import pytest

import ocnn
import ocnn.nn.kernels.config
from ocnn.octree import Points, Octree

# !!! disable TF32 for testing !!!
ocnn.nn.kernels.config.allow_tf32 = False


def sphere_coords(resolution, device="cuda"):
  r"""This function generates random features and integer coordinates for
  voxels on a thin spherical shell inside a cubic grid of resolution
  `res`. It iterates in n^3 chunks to keep memory bounded, building 3D
  meshes via `torch.meshgrid` and shifting them into global coordinates.

  Args:
    resolution: int
      The resolution of the cubic grid.
    device: str
      The device where the tensors are allocated.
  """

  n = 128
  out = []
  for i in range(0, resolution, n):
    for j in range(0, resolution, n):
      for k in range(0, resolution, n):
        block = torch.stack(
          torch.meshgrid(
            torch.arange(i, min(i + n, resolution), device=device),
            torch.arange(j, min(j + n, resolution), device=device),
            torch.arange(k, min(k + n, resolution), device=device),
            indexing="ij",
          ),
          dim=-1,
        ).int()
        dist = ((block.float() - resolution / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
        active = (dist <= resolution / 2) & (dist >= resolution / 2 - 1.25)
        out.append(block[active])
  pos = torch.cat(out, dim=0)
  return pos


def calc_err(src, ref):
    abs_err = (src - ref).float().abs()
    rel_err = abs_err / torch.clamp_min(ref.float().abs(), 1e-6)
    err = torch.minimum(abs_err, rel_err)
    return err.max().item(), err.mean().item()


# atol = 5e-3
device = "cuda"


@pytest.fixture(scope="module")
def test_octree():
    """创建用于测试的octree对象"""
    points = sphere_coords(64, device="cpu")
    points = points / 64 * 2 - 1
    octree = Octree(8, 2)
    octree.build_octree(Points(points))
    octree.construct_all_neigh()
    octree = octree.to(device)
    return octree


@pytest.mark.parametrize("depth", range(3, 9))
@pytest.mark.parametrize("out_ratio", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("channel", [2 ** i + 10 for i in range(8)] + [2 ** i + 5 for i in range(8)] + [2 ** i + 2 ** (i // 2) for i in range(8)])
def test_conv_stride1(depth, out_ratio, channel, test_octree, cleanup_cuda):
  kernel_size = [3]
  atol = 5e-3
  nempty = False
  dtype = torch.float32
  in_channel = channel
  out_channel = int(in_channel * out_ratio)
  conv_im2col = (
    ocnn.nn.OctreeConv(
      in_channel,
      out_channel,
      kernel_size,
      stride=1,
      nempty=nempty,
      direct_method=True,
      use_bias=True,
    )
    .type(dtype)
    .to(device)
  )
  conv_triton = (
    ocnn.nn.OctreeConvTriton(
      in_channel,
      out_channel,
      kernel_size,
      stride=1,
      nempty=nempty,
      use_bias=True,
    )
    .type(dtype)
    .to(device)
  )
  with torch.no_grad():
    conv_triton.weights.copy_(conv_im2col.weights)
    conv_triton.bias.copy_(conv_im2col.bias)
  data = torch.randn(test_octree.nnum[depth], in_channel, device=device, dtype=dtype)
  data_tt = data.detach().clone().requires_grad_()
  data_pt = data.detach().clone().requires_grad_()
  grad = torch.randn(
    test_octree.nnum[depth], out_channel, device=device, dtype=dtype
  )

  out_tt = conv_triton(data_tt, test_octree, depth)
  out_pt = conv_im2col(data_pt, test_octree, depth)

  loss_tt = (out_tt * grad).sum()
  loss_pt = (out_pt * grad).sum()
  loss_tt.backward()
  loss_pt.backward()

  assert torch.allclose(out_tt, out_pt, atol=atol), (
    "got output error rate {}".format(calc_err(out_tt, out_pt))
  )
  assert torch.allclose(
    conv_im2col.weights.grad,
    conv_triton.weights.grad,
    atol=atol,
  ), "got weight gradient error rate {}".format(
    calc_err(
      conv_im2col.weights.grad, conv_triton.weights.grad
    )
  )
  assert torch.allclose(
    conv_im2col.bias.grad, conv_triton.bias.grad, atol=atol
  ), "got bias gradient error rate {}".format(
    calc_err(conv_im2col.bias.grad, conv_triton.bias.grad)
  )
  assert torch.allclose(data_pt.grad, data_tt.grad, atol=atol), (
    "got input error rate {}".format(calc_err(data_pt.grad, data_tt.grad))
  )
