import os
import torch
import triton
import ocnn
import ocnn.nn.kernels
from ocnn.octree import Points, Octree
from utils import sphere_coords

#  pip install spconv-cu126
import spconv.pytorch as spconv


device = 'cuda'
depth2channel = {4: 512, 5: 256, 6: 128, 7: 128, 8: 64, 9: 32, 10: 32}
fp32_precision = 'ieee' if not ocnn.nn.kernels.config.allow_tf32 else 'tf32'


# Number-of-voxels at different depths:
#      depth      Number-of-voxels
# 0      5.0            3840
# 1      6.0           15192
# 2      7.0           64160
# 3      8.0          252392
# 4      9.0         1026536
# 5     10.0         4113056


configs = [
    triton.testing.Benchmark(
        x_names=['depth'],
        x_vals=[5, 6, 7, 8, 9, 10],
        line_arg='provider',
        line_vals=['torch', 'triton', 'spconv'],
        line_names=['OctConv', 'OctConvT', 'Spconv'],
        styles=[('green', '-'), ('red', '-'), ('blue', '-')],
        ylabel='Latency (ms)',
        plot_name=f'{fp32_precision}-{mode}-{str(dtype)}',
        args={'mode': mode, 'dtype': dtype},
        y_log=True,
    )
    for mode in ['fwd', 'bwd']
    for dtype in [torch.float32, torch.float16, torch.bfloat16]
]


@triton.testing.perf_report(configs)
def benchmark(depth, provider, mode, dtype):
  in_channel = depth2channel[depth]
  out_channel = in_channel
  kernel_size = [3, 3, 3]
  stride = 1
  nempty = True

  if provider == 'spconv' and dtype == torch.bfloat16:
    return float('nan')

  # Create modules
  if provider == 'torch':
    model = ocnn.nn.OctreeConv(
        in_channel, out_channel, kernel_size, stride=stride,
        nempty=nempty, use_bias=True,).type(dtype).to(device)

  elif provider == 'triton':
    model = ocnn.nn.OctreeConvTriton(
        in_channel, out_channel, kernel_size, stride=stride,
        nempty=nempty, use_bias=True,).type(dtype).to(device)
  else:
    model = spconv.SubMConv3d(
        in_channel, out_channel, kernel_size, padding=1, indice_key='test',
        algo=spconv.ConvAlgo.MaskSplitImplicitGemm, bias=True).cuda().to(dtype)
    model.weight.data.copy_(torch.randn(
        out_channel, *kernel_size, in_channel, dtype=dtype, device=device))
    model.bias.data.copy_(torch.randn(out_channel, dtype=dtype, device=device))

  # Generate coordinates and octree
  reso = 2**depth
  pos = sphere_coords(2**depth, device=device)
  pos = pos / reso * 2 - 1
  octree = Octree(depth, 2, device=device)
  octree.build_octree(Points(pos))
  octree.construct_all_neigh()
  nnum = octree.nnum_nempty[depth] if nempty else octree.nnum[depth]

  # Generate input data
  if provider == 'spconv':
    feat = torch.randn(
        nnum, in_channel, device=device, dtype=dtype, requires_grad=True,)
    x, y, z, b = octree.xyzb(depth, nempty=nempty)
    coords = torch.stack([b, x, y, z], dim=1).to(torch.int32)
    data = spconv.SparseConvTensor(
        feat, coords, torch.Size([reso, reso, reso]), 1)
    out_spconv = model(data)
    data.indice_dict = out_spconv.indice_dict.copy()
  else:
    data = torch.randn(
        nnum, in_channel, device=device, dtype=dtype, requires_grad=True,)

  if mode == 'fwd':
    if provider != 'spconv':

      def run_fwd():
        return model(data, octree, depth)
    else:

      def run_fwd():
        return model(data)

    ms = triton.testing.do_bench(run_fwd)

  else:
    if provider != 'spconv':
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


if __name__ == '__main__':
  curr_path = os.path.dirname(os.path.abspath(__file__))
  rst_path = os.path.join(curr_path, 'benchmark')
  os.makedirs(rst_path, exist_ok=True)
  benchmark.run(print_data=True, show_plots=False, save_path=rst_path)
