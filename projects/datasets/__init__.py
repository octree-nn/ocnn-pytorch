from .shapenet_seg import get_shapenet_seg_dataset
from .modelnet40 import get_modelnet40_dataset
from .scannet import get_scannet_dataset

__all__ = [
    'get_modelnet40_dataset',
    'get_shapenet_seg_dataset',
    'get_scannet_dataset',
]
