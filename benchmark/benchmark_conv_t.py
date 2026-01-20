import torch
import triton
import ocnn
import spconv.pytorch as spconv
from ocnn.octree import Points, Octree


def sphere_coords(resolution, device="cuda"):
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


device = "cuda"
depth2channel = {3: 1024, 4: 512, 5: 256, 6: 128, 7: 64, 8: 32}


configs = [
  triton.testing.Benchmark(
    x_names=["depth"],
    x_vals=[3, 4, 5, 6, 7, 8],
    line_arg="provider",
    line_vals=["torch", "triton", "spconv"],
    line_names=["PyTorch", "Triton", "Spconv"],
    styles=[("green", "-"), ("blue", "-"), ('yellow', '-')],
    ylabel="Latency (ms)",
    plot_name=f"ieee-{mode}-{str(dtype)}",
    args={"mode": mode, 'dtype': dtype},
  )
  for mode in ["fwd", 'bwd']
  for dtype in [torch.float32, torch.float16, torch.bfloat16]
]


@triton.testing.perf_report(configs)
def benchmark(depth, provider, mode, dtype):
  in_channel = depth2channel[depth]
  out_channel = in_channel
  kernel_size = [3]
  stride = 1
  nempty = False

  if provider == 'spconv' and dtype == torch.bfloat16:
    return float('nan')

  if provider == "torch":
    model = (
      ocnn.nn.OctreeConv(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        nempty=nempty,
        use_bias=True,
      )
      .type(dtype)
      .to(device)
    )
  elif provider == "triton":
    model = (
      ocnn.nn.OctreeConvTriton(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        nempty=nempty,
        use_bias=True,
      )
      .type(dtype)
      .to(device)
    )
  else:
    model = (
      spconv.SubMConv3d(
        in_channel,
        out_channel,
        (3, 3, 3),
        padding=1,
        indice_key="test",
        algo=spconv.ConvAlgo.MaskSplitImplicitGemm,
        bias=True
      )
      .cuda()
      .to(dtype)
    )
    model.weight.data.copy_(torch.randn(out_channel, 3, 3, 3, in_channel, dtype=dtype, device=device))
    model.bias.data.copy_(torch.randn(out_channel, dtype=dtype, device=device))

  reso = 2**depth
  coords = sphere_coords(2**depth)
  if provider == "spconv":
    feat = torch.randn(
      len(coords),
      in_channel,
      device=device,
      dtype=dtype,
      requires_grad=True,
    )
    coords = torch.cat([torch.zeros(coords.shape[0], 1, device=device, dtype=torch.int32), coords], dim=-1)
    data = spconv.SparseConvTensor(
      feat, coords, torch.Size([reso, reso, reso]), 1
    )
    out_spconv = model(data)
    data.indice_dict = out_spconv.indice_dict.copy()
  else:
    pos = coords / reso * 2 - 1
    octree = Octree(depth, 2, device=device)
    octree.build_octree(Points(pos))
    octree.construct_all_neigh()
    data = torch.randn(
      octree.nnum[depth],
      in_channel,
      device=device,
      dtype=dtype,
      requires_grad=True,
    )

  if mode == "fwd":
    if provider != "spconv":

      def run_fwd():
        return model(data, octree, depth)
    else:

      def run_fwd():
        return model(data)

    ms = triton.testing.do_bench(run_fwd)

  else:
    if provider != "spconv":
      out = model(data, octree, depth)
      grad_out = torch.randn_like(out)
      def run_bwd():
        out.backward(grad_out, retain_graph=True)
    else:
      out = model(data)
      grad_out = torch.randn_like(out.features)
      def run_bwd():
        out.features.backward(grad_out, retain_graph=True)

    ms = triton.testing.do_bench(run_bwd)

  return ms


if __name__ == "__main__":
  benchmark.run(print_data=True, show_plots=False, save_path=".")
