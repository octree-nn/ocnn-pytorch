import torch
import numpy as np
from typing import Optional, Union, List


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
    batch_size (int): The batch size.
  '''

  def __init__(self, points: torch.Tensor,
               normals: Optional[torch.Tensor] = None,
               features: Optional[torch.Tensor] = None,
               labels: Optional[torch.Tensor] = None,
               batch_size: int = 1):
    self.points = points
    self.normals = normals
    self.features = features
    self.labels = labels
    self.device = points.device
    self.batch_size = batch_size
    self.batch_npt = None
    self.batch_id = None

  def orient_normal(self, axis: str = 'x'):
    r''' Orients the point normals along a given axis.

    Args:
      axis (int): The coordinate axes, choose from :obj:`x`, :obj:`y` and 
          :obj:`z`. (default: :obj:`x`)
    '''

    if self.normals is None:
      return

    axis_map = {'x': 0, 'y': 1, 'z': 2, 'xyz': 3}
    idx = axis_map[axis]
    if idx < 3:
      flags = self.normals[:, idx] > 0
      flags = flags.float() * 2.0 - 1.0  # [0, 1] -> [-1, 1]
      self.normals = self.normals * flags.unsqueeze(1)
    else:
      self.normals.abs_()

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

    factor = factor.to(self.device)
    self.points = self.points * factor
    if self.normals is not None and non_uniform:
      ifactor = 1.0 / factor
      self.normals = self.normals * ifactor
      norm2 = torch.sqrt(torch.sum(self.normals ** 2, dim=1, keepdim=True))
      self.normals = self.normals / torch.clamp(norm2, min=1.0e-12)

  def rotate(self, angle: torch.Tensor):
    r''' Rotates the point cloud. 

    Args:
      angle (torch.Tensor): The rotation angles in radian with shape :obj:`(3,)`.
    '''

    cos, sin = angle.cos(), angle.sin()
    # rotx, roty, rotz are actually the transpose of the rotation matrices
    rotx = torch.Tensor([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
    roty = torch.Tensor([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
    rotz = torch.Tensor([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])
    rot = rotx @ roty @ rotz

    rot = rot.to(self.device)
    self.points = self.points @ rot
    if self.normals is not None:
      self.normals = self.normals @ rot

  def translate(self, dis: torch.Tensor):
    r''' Translates the point cloud. 

    Args:
      dis (torch.Tensor): The displacement with shape :obj:`(3,)`.
    '''

    dis = dis.to(self.device)
    self.points = self.points + dis

  def flip(self, axis: str):
    r''' Flips the point cloud along the given :attr:`axis`.

    Args:
      axis (str): The flipping axis, choosen from :obj:`x`, :obj:`y`, and :obj`z`.
    '''

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    idx = axis_map[axis]
    self.points[:, idx] *= -1.0
    if self.normals is not None:
      self.normals[:, idx] *= -1.0

  def clip(self, min: float = -1.0, max: float = 1.0, esp: float = 0.01):
    r''' Clips the point cloud to :obj:`[min+esp, max-esp]` and returns the mask.

    Args:
      min (float): The minimum value to clip.
      max (float): The maximum value to clip.
      esp (float): The margin.
    '''

    mask_min = torch.all(self.points > min + esp, dim=1)
    mask_max = torch.all(self.points < max - esp, dim=1)
    mask = torch.logical_and(mask_min, mask_max)

    self.points = self.points[mask]
    if self.normals is not None:
      self.normals = self.normals[mask]
    if self.features is not None:
      self.features = self.features[mask]
    if self.labels is not None:
      self.labels = self.labels[mask]
    if self.batch_id is not None:
      self.batch_id = self.batch_id[mask]
      self.batch_npt = None  # TODO: Update batch_npt
    return mask

  def bbox(self):
    r''' Returns the bounding box.
    '''

    # torch.min and torch.max return (value, indices)
    bbmin = self.points.min(dim=0)
    bbmax = self.points.max(dim=0)
    return bbmin[0], bbmax[0]

  def normalize(self, bbmin: torch.Tensor, bbmax: torch.Tensor, scale: float = 1.0):
    r''' Normalizes the point cloud to :obj:`[-scale, scale]`.

    Args:
      bbmin (torch.Tensor): The minimum coordinates of the bounding box.
      bbmax (torch.Tensor): The maximum coordinates of the bounding box.
      scale (float): The scale factor
    '''

    center = (bbmin + bbmax) * 0.5
    box_size = (bbmax - bbmin).max() + 1.0e-6
    self.points = (self.points - center) * (2.0 * scale / box_size)

  def to(self, device: Union[torch.device, str]):
    r''' Moves the Points to a specified device. 

    Args:
      device (torch.device or str): The destination device.
    '''

    if isinstance(device, str):
      device = torch.device(device)

    #  If on the save device, directly retrun self
    if self.device == device:
      return self

    # Construct a new Points on the specified device
    points = Points(torch.zeros(1, 3, device=device))
    points.batch_npt = self.batch_npt
    points.points = self.points.to(device)
    if self.normals is not None:
      points.normals = self.normals.to(device)
    if self.features is not None:
      points.features = self.features.to(device)
    if self.labels is not None:
      points.labels = self.labels.to(device)
    if self.batch_id is not None:
      points.batch_id = self.batch_id.to(device)
    return points

  def cuda(self):
    r''' Moves the Points to the GPU. '''

    return self.to('cuda')

  def cpu(self):
    r''' Moves the Points to the CPU. '''

    return self.to('cpu')

  def save(self, filename: str, save_batch: bool = False):
    r''' Save the Points into npz or xyz files.

    Args:
      filename (str): The output filename.
      save_batch (bool): Whether to save the batch index.
    '''

    name = ['points']
    out = [self.points.cpu().numpy()]
    if self.normals is not None:
      name.append('normals')
      out.append(self.normals.cpu().numpy())
    if self.features is not None:
      name.append('features')
      out.append(self.features.cpu().numpy())
    if self.labels is not None:
      name.append('labels')
      labels = self.labels
      if labels.dim() == 1:
        labels = labels.unsqueeze(1)
      out.append(labels.cpu().numpy())
    if self.batch_id is not None and save_batch:
      name.append('batch_id')
      batch_id = self.batch_id
      if batch_id.dim() == 1:
        batch_id = batch_id.unsqueeze(1)
      out.append(batch_id.cpu().numpy())

    if filename.endswith('npz'):
      out_dict = dict(zip(name, out))
      np.savez(filename, **out_dict)
    elif filename.endswith('xyz'):
      out_array = np.concatenate(out, axis=1)
      np.savetxt(filename, out_array, fmt='%.6f')
    else:
      raise ValueError


def merge_points(points: List['Points']):
  r''' Merges a list of points into one batch.

  Args:
    points (List[Octree]): A list of points to merge.
  '''

  out = Points(torch.zeros(1, 3))
  out.points = torch.cat([p.points for p in points], dim=0)
  if points[0].normals is not None:
    out.normals = torch.cat([p.normals for p in points], dim=0)
  if points[0].features is not None:
    out.features = torch.cat([p.features for p in points], dim=0)
  if points[0].labels is not None:
    out.labels = torch.cat([p.labels for p in points], dim=0)
  out.device = points[0].device
  out.batch_size = len(points)
  out.batch_npt = torch.Tensor([p.points.shape[0] for p in points])
  out.batch_id = torch.cat([p.points.new_full((p.points.shape[0], 1), i)
                            for i, p in enumerate(points)], dim=0)
  return out
