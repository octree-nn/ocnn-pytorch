import torch
import pytest
import copy
import ocnn
from ocnn.octree import Points, Octree
from ocnn.modules import OctreeResBlocks, OctreeResBlock, OctreeResBlock2, OctreeResBlockGn
from ocnn.models import ResNet


def calc_err(src, ref):
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


ocnn.nn.kernels.config.allow_tf32 = False
atol = 5e-3
device = "cuda"
points = sphere_coords(64, device="cpu")
points = points / 64 * 2 - 1
octree1 = Octree(8, 2)
octree1.build_octree(Points(points))
octree2 = Octree(8, 2)
octree2.build_octree(Points(points))
octree = ocnn.octree.merge_octrees([octree1, octree2])  # batch size = 2 for bn in ResNet
octree.construct_all_neigh()
octree = octree.to(device)
depth2channel = {3: 1024, 4: 512, 5: 256, 6: 128, 7: 64, 8: 32}


@pytest.mark.parametrize("depth", range(3, 9))
@pytest.mark.parametrize("dtype", [torch.float32, ])
# @pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("nempty", [False, True])
@pytest.mark.parametrize("use_bias", [True, False])
def test_convert_conv(depth: int, dtype: torch.dtype, nempty: bool, use_bias: bool):
    in_channel = depth2channel[depth]
    out_channel = in_channel
    data = torch.randn(octree.nnum[depth] if not nempty else octree.nnum_nempty[depth], in_channel, device=device, dtype=dtype)
    conv = ocnn.nn.OctreeConv(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=[3, 3, 3],
        stride=1,
        nempty=nempty,
        use_bias=use_bias,
    ).cuda().type(dtype)
    activation1 = conv(data, octree, depth)
    conv_triton = ocnn.nn.utils.convert_conv_triton(conv)
    assert isinstance(conv_triton, ocnn.nn.OctreeConvTriton)
    activation2 = conv_triton(data, octree, depth)
    assert torch.allclose(activation1, activation2, atol=atol)


@pytest.mark.parametrize("depth", range(3, 9))
@pytest.mark.parametrize("resblk", [OctreeResBlock, OctreeResBlock2, OctreeResBlockGn])
@pytest.mark.parametrize("nempty", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, ])
def test_convert_module(depth: int, resblk: torch.nn.Module, nempty: bool, dtype: torch.dtype):
    in_channel = depth2channel[depth]
    out_channel = in_channel
    module = OctreeResBlocks(in_channel, out_channel, 2, resblk=resblk, nempty=nempty)
    module = module.cuda().type(dtype)  
    data = torch.randn(octree.nnum[depth] if not nempty else octree.nnum_nempty[depth], in_channel, device=device, dtype=dtype)
    activation1 = module(data, octree, depth)
    module_triton = ocnn.nn.utils.convert_conv_triton(module)
    activation2 = module_triton(data, octree, depth)
    assert torch.allclose(activation1, activation2, atol=atol), (calc_err(activation1, activation2))


@pytest.mark.parametrize("depth", range(5, 9))
@pytest.mark.parametrize("dtype", [torch.float32, ])
@pytest.mark.parametrize("nempty", [False, True])
@pytest.mark.parametrize("stages", range(1, 4))
def test_convert_model(depth: int, dtype: torch.dtype, nempty: bool, stages: int):
    in_channel = depth2channel[depth]
    out_channel = 1000
    model = ResNet(in_channel, out_channel, 2, stages, nempty=nempty)
    model.header = torch.nn.Sequential(
        ocnn.modules.FcBnRelu(model.header[0].fc.in_features, 512),
        torch.nn.Linear(512, out_channel))  # remove dropout for testing
    model = model.cuda().type(dtype)
    data = torch.randn(octree.nnum[depth] if not nempty else octree.nnum_nempty[depth], in_channel, device=device, dtype=dtype)
    activation1 = model(data, octree, depth)
    model_triton = ocnn.nn.utils.convert_conv_triton(model)
    assert isinstance(model_triton, ResNet)
    activation2 = model_triton(data, octree, depth)
    assert torch.allclose(activation1, activation2, atol=atol), calc_err(activation1, activation2)