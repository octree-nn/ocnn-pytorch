from .lenet import LeNet
from .resnet import ResNet
from .segnet import SegNet
from .unet import UNet
from .hrnet import HRNet

__all__ = [
    'LeNet',
    'ResNet',
    'SegNet',
    'UNet',
    'HRNet'
]

classes = __all__
