"""Render source / Triton / cubvh meshes with shared cameras and material.

Uses EGL on the GPU selected by EGL_DEVICE_ID. Read comparison results from
compare_sparse_mc.py; writes full-resolution PNG panels and contact sheets.
"""

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrender
import trimesh


NAMES = {'02691156': 'Airplane', '03001627': 'Chair', '02958343': 'Car'}


def pose_for(eye, center):
  backward = np.asarray(eye, float) - center
  backward /= np.linalg.norm(backward)
  right = np.cross([0., 1., 0.], backward)
  right /= np.linalg.norm(right)
  up = np.cross(backward, right)
  pose = np.eye(4)
  pose[:3, :3] = np.stack((right, up, backward), 1)
  pose[:3, 3] = eye
  return pose


def camera_for(meshes, direction, aspect):
  bounds = np.array([m.bounds for m in meshes])
  low, high = bounds[:, 0].min(0), bounds[:, 1].max(0)
  center = (low + high) / 2
  direction = np.asarray(direction, float)
  pose = pose_for(center + direction / np.linalg.norm(direction) * 4, center)
  box = np.array(list(itertools.product(*zip(low, high))))
  projected = (box - center) @ pose[:3, :3]
  ymag = max(np.abs(projected[:, 1]).max(),
              np.abs(projected[:, 0]).max()/aspect) * 1.08
  return pyrender.OrthographicCamera(xmag=ymag*aspect, ymag=ymag), pose


def draw_mesh(renderer, mesh, camera, pose):
  scene = pyrender.Scene(bg_color=[0.975, 0.981, 0.989, 1],
                         ambient_light=[0.43, 0.43, 0.43])
  material = pyrender.MetallicRoughnessMaterial(
      baseColorFactor=[0.38, 0.55, 0.70, 1], metallicFactor=0.,
      roughnessFactor=.75, doubleSided=True)
  scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
  scene.add(camera, pose=pose)
  for direction, intensity in (([3, 4, 3], 2.8), ([-3, 1, 2], 1.5),
                                ([0, 3, -3], 1.8)):
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
              pose=pose_for(np.asarray(direction), np.zeros(3)))
  color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
  return Image.fromarray(color[:, :, :3])


def fonts():
  from matplotlib.font_manager import findfont
  path = findfont('DejaVu Sans')
  return {size: ImageFont.truetype(path, size) for size in (18, 21, 24, 30, 38)}


def contact_sheet(rows, view, target, font):
  panel_w, panel_h, gutter = 640, 480, 20
  width = panel_w*3 + gutter*4
  header, row_h = 155, panel_h+85
  sheet = Image.new('RGB', (width, header+len(rows)*row_h+55), '#ffffff')
  draw = ImageDraw.Draw(sheet)
  draw.text((gutter, 18), 'ShapeNet r256 | source mesh vs sparse Marching Cubes',
             font=font[38], fill='#13283d')
  titles = ['Source OBJ (normalized)', 'Octree SDF / Triton', 'Octree SDF / cubvh']
  for col, title in enumerate(titles):
    draw.text((gutter+(panel_w+gutter)*col, 95), title,
               font=font[30], fill='#203c56')
  for row, item in enumerate(rows):
    y = header + row*row_h
    name = NAMES.get(item['sample'].split('/')[0], 'Shape')
    if item['sample'].endswith('88ad6f5df57cf8ea16abce8cb03e7794'):
      name += ' / repaired SDF case'
    draw.text((gutter, y), name+'  '+item['sample'].split('/')[1][:12],
               font=font[24], fill='#233b51')
    for col, kind in enumerate(('source', 'triton', 'cubvh')):
      x = gutter + col*(panel_w+gutter)
      image = Image.open(item['panels'][view][kind]).convert('RGB')
      sheet.paste(image.resize((panel_w, panel_h), Image.Resampling.LANCZOS),
                   (x, y+37))
      description = f"{item[kind+'_faces']:,} triangles"
      if kind != 'source':
        timing = 'ours_octree_total' if kind == 'triton' else 'cubvh_octree_total'
        description += f" | {item['timings'][timing]['median_ms']:.2f} ms from octree"
      draw.text((x, y+panel_h+43), description, font=font[21], fill='#465b6f')
  draw.text((gutter, sheet.height-35),
             'Shared camera and clay material; source aligned using r256/bbox metadata.',
             font=font[18], fill='#63788b')
  sheet.save(target)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('results', type=Path)
  args = parser.parse_args()
  report = json.loads((args.results/'report.json').read_text())
  renderer = pyrender.OffscreenRenderer(1280, 960)
  rows = []
  try:
    for item in report['samples']:
      directory = args.results/item['sample']
      meshes = {kind: trimesh.load(directory/(kind+'.ply'), process=False)
                 for kind in ('source', 'triton', 'cubvh')}
      # PLY may retain source OBJ vertex/face colors. Neutralize those too,
      # so every column really uses the same clay material.
      for mesh in meshes.values():
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh, vertex_colors=np.tile([255, 255, 255, 255],
                                             (len(mesh.vertices), 1)))
      item['panels'] = {}
      for view, direction in (('front', [2.5, 1.6, 2.4]),
                               ('reverse', [-2.5, 1.6, -2.4])):
        camera, pose = camera_for(list(meshes.values()), direction, 4/3)
        item['panels'][view] = {}
        for kind, mesh in meshes.items():
          image = draw_mesh(renderer, mesh, camera, pose)
          path = directory/f'{kind}-{view}.png'
          image.save(path)
          item['panels'][view][kind] = str(path)
          item[kind+'_faces'] = len(mesh.faces)
      rows.append(item)
  finally:
    renderer.delete()
  font = fonts()
  for view in ('front', 'reverse'):
    path = args.results/f'side-by-side-{view}.png'
    contact_sheet(rows, view, path, font)
    print(path, flush=True)


if __name__ == '__main__':
  main()
