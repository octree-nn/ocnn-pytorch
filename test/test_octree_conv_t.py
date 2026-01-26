# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import pytest
import unittest

import ocnn
import ocnn.nn.kernels.config
from ocnn.octree import Points, Octree
# from .utils import sphere_coords

# !!! disable TF32 for testing !!!
ocnn.nn.kernels.config.allow_tf32 = False


class TestOctreeConvTriton(unittest.TestCase):

  def test_conv_forward_backward(self):
    if torch.cuda.is_available() is False:
      return  # skip test if no GPU

    r = 64
    depth, full_depth = 7, 3
    pos = sphere_coords(64, device='cuda')
    pos = pos / r * 2.0 - 1.0  # normalize to [-1,1]
    points = Points(points=pos)
    octree = Octree(depth, full_depth, device='cuda')
    octree.build_octree(points)
    octree.construct_all_neigh()

    self.depth2channel = {3: 1024, 4: 512, 5: 256, 6: 128, 7: 64}
    for d in range(depth - 1, depth + 1):
      for out_ratio in [1.0, 0.5, 2.0]:
        self.conv_forward_backward(d, out_ratio, octree)

  def conv_forward_backward(self, depth, out_ratio, octree):
    atol = 5e-3
    kernel_size = [3]
    nempty = False
    device = 'cuda'
    in_channel = self.depth2channel[depth]
    out_channel = int(in_channel * out_ratio)

    # initialize conv layers
    conv_pt = ocnn.nn.OctreeConv(
        in_channel, out_channel, kernel_size, stride=1,
        nempty=nempty, use_bias=True).to(device)
    conv_tt = ocnn.nn.OctreeConvTriton(
        in_channel, out_channel, kernel_size, stride=1,
        nempty=nempty, use_bias=True,).to(device)
    with torch.no_grad():
      conv_tt.weights.copy_(conv_pt.weights)
      conv_tt.bias.copy_(conv_pt.bias)

    # initialize data and grad
    data = torch.randn(octree.nnum[depth], in_channel, device=device)
    data_tt = data.detach().clone().requires_grad_()
    data_pt = data.detach().clone().requires_grad_()
    grad = torch.randn(octree.nnum[depth], out_channel, device=device)

    # forward
    out_tt = conv_tt(data_tt, octree, depth)
    out_pt = conv_pt(data_pt, octree, depth)

    # backward
    loss_tt = (out_tt * grad).sum()
    loss_pt = (out_pt * grad).sum()
    loss_tt.backward()
    loss_pt.backward()

    # check results
    msg = f'depth: {depth}, out_ratio: {out_ratio}'
    self.assertTrue(torch.allclose(out_tt, out_pt, atol=atol), msg)
    self.assertTrue(torch.allclose(data_pt.grad, data_tt.grad, atol=atol), msg)
    self.assertTrue(torch.allclose(
        conv_pt.weights.grad, conv_tt.weights.grad, atol=atol), msg)
    self.assertTrue(torch.allclose(
        conv_pt.bias.grad, conv_tt.bias.grad, atol=atol), msg)

  def calc_err(self, src, ref):
    abs_err = (src - ref).float().abs()
    rel_err = abs_err / torch.clamp_min(ref.float().abs(), 1e-6)
    err = torch.minimum(abs_err, rel_err)
    return err.max().item(), err.mean().item()


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


# atol = 5e-3
device = "cuda"
points = sphere_coords(64, device="cpu")
points = points / 64 * 2 - 1
octree = Octree(8, 2)
octree.build_octree(Points(points))
octree.construct_all_neigh()
octree = octree.to(device)
depth2channel = {3: 1024, 4: 512, 5: 256, 6: 128, 7: 64, 8: 32}


@pytest.mark.parametrize("depth", range(3, 9))
@pytest.mark.parametrize("out_ratio", [1.0, 0.5, 2.0])
def test_conv_stride1(depth, out_ratio):
  kernel_size = [3]
  atol = 5e-3
  nempty = False
  dtype = torch.float32
  in_channel = depth2channel[depth]
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
  data = torch.randn(octree.nnum[depth], in_channel, device=device, dtype=dtype)
  data_tt = data.detach().clone().requires_grad_()
  data_pt = data.detach().clone().requires_grad_()
  grad = torch.randn(
    octree.nnum[depth], out_channel, device=device, dtype=dtype
  )

  out_tt = conv_triton(data_tt, octree, depth)
  out_pt = conv_im2col(data_pt, octree, depth)

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


def run_error_rate():
  headers = [
    "Dtype",
    "Depth",
    "Fwd Max",
    "Fwd Mean",
    "W.Grad Max",
    "W.Grad Mean",
    "B.Grad Max",
    "B.Grad Mean",
    "In.Grad Max",
    "In.Grad Mean",
    "Status",
  ]

  # 打印表头 (设置列宽)
  header_fmt = "{:<16} | {:<5} | {:<10} {:<10} | {:<12} {:<12} | {:<12} {:<12} | {:<12} {:<12} | {:<6}"
  print("-" * 140)
  print(header_fmt.format(*headers))
  print("-" * 140)

  # 扫描参数
  dtypes = [torch.float16, torch.bfloat16, torch.float32]
  depths = range(3, 8)

  for dtype in dtypes:
    for depth in depths:
      try:
        out_ratio = 1
        kernel_size = [3, 3, 3]
        nempty = False
        in_channel = depth2channel[depth]
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

        num_points = octree.nnum[depth]
        data = torch.randn(num_points, in_channel, device=device, dtype=dtype)

        data_tt = data.detach().clone().requires_grad_()
        data_im2col = data.detach().clone().requires_grad_()
        grad = torch.randn(num_points, out_channel, device=device, dtype=dtype)

        out_tt = conv_triton(data_tt, octree, depth)
        out_im2col = conv_im2col(data_im2col, octree, depth)

        loss_tt = (out_tt * grad).sum()
        loss_im2col = (out_im2col * grad).sum()
        loss_tt.backward()
        loss_im2col.backward()

        tt_err = dict()
        tt_err["fwd"] = calc_err(out_tt, out_im2col)
        tt_err["bwd_w"] = calc_err(
          conv_triton.weights.grad, conv_im2col.weights.grad
        )
        tt_err["bwd_b"] = calc_err(conv_triton.bias.grad, conv_im2col.bias.grad)
        tt_err["bwd_in"] = calc_err(data_tt.grad, data_im2col.grad)

        failed = False
        for k, v in tt_err.items():
          if v[1] >= 15e-3:  # mean error check
            failed = True

        dtype_str = str(dtype)
        status = "FAIL" if failed else "SUCC"

        def fmt(val):
          return f'{val * 1000:.0f}‰'

        row_data = [
          dtype_str,
          depth,
          fmt(tt_err["fwd"][0]),
          fmt(tt_err["fwd"][1]),
          fmt(tt_err["bwd_w"][0]),
          fmt(tt_err["bwd_w"][1]),
          fmt(tt_err["bwd_b"][0]),
          fmt(tt_err["bwd_b"][1]),
          fmt(tt_err["bwd_in"][0]),
          fmt(tt_err["bwd_in"][1]),
          status,
        ]

        print(header_fmt.format(*row_data))

      except Exception as e:
        dtype_str = "fp16" if dtype == torch.float16 else "bf16"
        print(f"{dtype_str:<10} | {depth:<5} | Error: {str(e)}")
        import traceback

        traceback.print_exc()

if __name__ == '__main__':
  run_error_rate()