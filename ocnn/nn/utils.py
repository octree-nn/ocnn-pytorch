import torch


from .octree_conv import OctreeConv
from .octree_conv_t import OctreeConvTriton


def convert_conv_triton(module):
    module_out = module
    if isinstance(module, OctreeConv) and module.stride == 1 and module.kernel_size == [3, 3, 3]:
        module_out = OctreeConvTriton(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.nempty,
            use_bias=module.use_bias,
        )
        with torch.no_grad():
            module_out.weights = module.weights
            if module.use_bias:
                module_out.bias = module.bias
    for name, child in module.named_children():
        module_out.add_module(name, convert_conv_triton(child))
    del module
    return module_out
