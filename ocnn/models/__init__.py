from .lenet import LeNet
from .resnet import ResNet
from .segnet import SegNet
from .unet import UNet
from .hrnet import HRNet
from .autoencoder import AutoEncoder

__all__ = [
    'LeNet',
    'ResNet',
    'SegNet',
    'UNet',
    'HRNet',
    'AutoEncoder',
]

classes = __all__
