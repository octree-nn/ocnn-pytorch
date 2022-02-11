import torch
import numpy as np
from typing import Optional, Union


class Points:
  r''' Represents a point cloud and contains some elementary transformations.

  Args:
    points (torch.Tensor): The coordinates of the points with a shape of 
        :obj:`(N, 3)`, where :obj:`N` is the number of points.
    normals (torch.Tensor or None): The point normals with a shape of
        :obj:`(N, 3)`.
    features (torch.Tensor or None): The point features with a shape of
        :obj:`(N, C)`, where :obj:`C` is the channel of features.
    labels (torch.Tensor or None): The point labels with a shape of
        :obj:`(N, K)`, where :obj:`K` is the channel of labels.
  '''

  def __init__(self, points: torch.Tensor,
               normals: Optional[torch.Tensor] = None,
               features: Optional[torch.Tensor] = None,
               labels: Optional[torch.Tensor] = None):
    self.points = points
    self.normals = normals
    self.features = features
    self.labels = labels
    self.npt = points.shape[0]

  def orient_normal(self, axis: str = 'x'):
    r''' Orients the point normals along a given axis.

    Args:
      axis (int): The coordinate axes, choose from :obj:`x`, :obj:`y` and 
          :obj:`z`. (default: :obj:`x`)
    '''

    if self.normals is None: return
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    idx = axis_map[axis]

    flags = self.normals[:, idx] > 0
    flags = flags.float() * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    self.normals = self.normals * flags.unsqueeze(1)

  def scale(self, factor: torch.Tensor):
    r''' Rescales the point cloud. 

    Args:
      factor (torch.Tensor): The scale factor with shape :obj:`(3,)`.
    '''

    non_zero = (factor != 0).all()
    all_ones = (factor == 1.0).all()
    non_uniform = (factor != factor[0]).any()
    torch._assert(non_zero, 'The scale factor must not constain 0.')
    if all_ones: return

    factor = factor.to(self.points.device)
    self.points = self.points * factor
    if self.normals is not None and non_uniform:
      ifactor = 1.0 / factor
      self.normals = self.normals * ifactor
      norm2 = torch.sqrt(torch.sum(self.normals ** 2, dim=1, keepdim=True))
      self.normals = self.normals / (norm2 + 1.0e-6)

  def rotate(self, angle: torch.Tensor):
    r''' Rotates the point cloud. 

    Args:
      angle (torch.Tensor): The rotation angles in radian with shape :obj:`(3,)`.
    '''

    cos, sin = angle.cos(), angle.sin()
    rotx = torch.Tensor([1, 0, 0, 0, cos[0], sin[0], 0, -sin[0], cos[0]])
    roty = torch.Tensor([cos[1], 0, -sin[1], 0, 1, 0, sin[1], 0, cos[1]])
    rotz = torch.Tensor([cos[2], sin[2], 0, -sin[2], cos[2], 0, 0, 0, 1])
    rot = rotx @ roty @ rotz

    rot = rot.to(self.points.device).t()
    self.points = self.points @ rot
    if self.normals is not None:
      self.normals = self.normals @ rot

  def translate(self, dis: torch.Tensor):
    r''' Translates the point cloud. 

    Args:
      dis (torch.Tensor): The displacement with shape :obj:`(3,)`.
    '''

    dis = dis.to(self.points.device)
    self.points = self.points + dis

  def clip(self, min: float = -1.0, max: float = 1.0):
    r''' Clips the point cloud to :obj:`[min, max]` and returns the mask.

    Args:
      min (float): The minimum value to clip.
      max (float): The maximum value to clip.
    '''

    mask_min = torch.all(self.points > min, dim=1)
    mask_max = torch.all(self.points < max, dim=1)
    mask = torch.logical_and(mask_min, mask_max)

    self.points = self.points[mask]
    if self.normals is not None:
      self.normals = self.normals[mask]
    if self.features is not None:
      self.features = self.features[mask]
    if self.labels is not None:
      self.labels = self.labels[mask]
    return mask

  def bbox(self):
    r''' Returns the bounding box.
    '''

    bbmin = self.points.min(dim=1)
    bbmax = self.points.max(dim=1)
    return bbmin, bbmax

  def normalize(self, bbmin: torch.Tensor, bbmax: torch.Tensor):
    r''' Normalizes the point cloud to :obj:`[-1, 1]`.

    Args:
      bbmin (torch.Tensor): The minimum coordinates of the bounding box.
      bbmax (torch.Tensor): The maximum coordinates of the bounding box.
    '''
    box_size = (bbmax - bbmin).max() + 1.0e-6
    self.points = (self.points - bbmin) * (2.0 / box_size) - 1.0

  def to(self, device: Union[torch.device, str]):
    r''' Moves the Points to a specified device. 

    Args:
      device (torch.device or str): The destination device.
    '''

    self.points = self.points.to(device)
    if self.normals is not None:
      self.normals = self.normals.to(device)
    if self.features is not None:
      self.features = self.features.to(device)
    if self.labels is not None:
      self.labels = self.labels.to(device)

  def save(self, filename: str):
    r''' Save the Points into npz or xyz files.

    Args:
      filename (str): The output filename.
    '''

    out = {'points': self.points.cpu().numpy()}
    if self.normals is not None:
      out['normals'] = self.normals.cpu().numpy()
    if self.features is not None:
      out['features'] = self.features.cpu().numpy()
    if self.labels is not None:
      out['labels'] = self.labels.cpu().numpy()

    if filename.endswith('npz'):
      np.savez(filename, **out)
    elif filename.endswith('xyz'):
      out = torch.cat(list(out.values()), dim=1)
      np.savetxt(filename, out, fmt='%.6f')
    else:
      raise ValueError
