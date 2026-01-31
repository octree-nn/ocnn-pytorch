import os
import torch
import triton
import ocnn
from ocnn.octree import Octree, Points
from utils import sphere_coords


device = 'cuda'
depth2channel = {3: 1024, 4: 512, 5: 256, 6: 128, 7: 64, 8: 32, 9: 16}
fp32_precision = 'ieee' if not ocnn.nn.kernels.config.allow_tf32 else 'tf32'


device = 'cuda'
points = sphere_coords(64, device='cpu')
points = points / 64 * 2 - 1
octree1 = Octree(8, 2)
octree1.build_octree(Points(points))
octree2 = Octree(8, 2)
octree2.build_octree(Points(points))
# batch size = 2 for bn in ResNet
octree = ocnn.octree.merge_octrees([octree1, octree2])
octree.construct_all_neigh()
octree = octree.to(device)
depth2channel = {3: 1024, 4: 512, 5: 256, 6: 128, 7: 64, 8: 32}


configs = [
    triton.testing.Benchmark(
        x_names=['depth'],
        x_vals=[5, 6, 7, 8],
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['OctConv', 'OctConvT'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel='Latency (ms)',
        plot_name=f'ResNet-{mode}-{fp32_precision}-{str(dtype)}',
        args={'dtype': dtype, 'mode': mode},
    )
    for dtype in [torch.float32, torch.float16, torch.bfloat16]
    for mode in ['fwd', 'bwd']
]


@triton.testing.perf_report(configs)
def benchmark_resnet(depth, provider, dtype, mode):
  in_channel = depth2channel[depth]
  out_channel = 10
  nempty = False
  model = ocnn.models.ResNet(in_channel, out_channel, 2, 4, nempty=nempty)
  model = model.cuda().type(dtype)
  num = octree.nnum[depth] if not nempty else octree.nnum_nempty[depth]

  if provider == 'triton':
    model = ocnn.nn.convert_conv_triton(model)
  if mode == 'fwd':
    data = torch.randn(num, in_channel, device=device, dtype=dtype)
  else:
    data = torch.randn(num, in_channel, device=device, dtype=dtype)
    out = model(data, octree, depth)
    grad_out = torch.randn_like(out)

  def run_fwd():
    out = model(data, octree, depth)
    return out

  def run_bwd():
    out.backward(grad_out, retain_graph=True)
    return out
  if mode == 'fwd':
    ms = triton.testing.do_bench(run_fwd)
  else:
    ms = triton.testing.do_bench(run_bwd)
  return ms


if __name__ == '__main__':
  curr_path = os.path.dirname(os.path.abspath(__file__))
  rst_path = os.path.join(curr_path, 'benchmark')
  os.makedirs(rst_path, exist_ok=True)
  benchmark_resnet.run(print_data=True, show_plots=False, save_path=rst_path)
