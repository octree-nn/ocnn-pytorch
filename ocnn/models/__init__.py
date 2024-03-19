# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

from .lenet import LeNet
from .resnet import ResNet
from .segnet import SegNet
from .unet import UNet
from .hrnet import HRNet
from .autoencoder import AutoEncoder
from .ounet import OUNet
from .image2shape import Image2Shape


__all__ = [
    'LeNet',
    'ResNet',
    'SegNet',
    'UNet',
    'HRNet',
    'AutoEncoder',
    'OUNet',
    'Image2Shape',
]

classes = __all__
