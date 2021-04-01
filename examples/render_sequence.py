# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of rendering a sequence."""

import argparse
import torch
import pyrender
import trimesh
import os
import numpy as np
import cv2

from dex_ycb_toolkit.sequence_loader import SequenceLoader

_YCB_COLORS = {
     1: (255,   0,   0),  # 002_master_chef_can
     2: (  0, 255,   0),  # 003_cracker_box
     3: (  0,   0, 255),  # 004_sugar_box
     4: (255, 255,   0),  # 005_tomato_soup_can
     5: (255,   0, 255),  # 006_mustard_bottle
     6: (  0, 255, 255),  # 007_tuna_fish_can
     7: (128,   0,   0),  # 008_pudding_box
     8: (  0, 128,   0),  # 009_gelatin_box
     9: (  0,   0, 128),  # 010_potted_meat_can
    10: (128, 128,   0),  # 011_banana
    11: (128,   0, 128),  # 019_pitcher_base
    12: (  0, 128, 128),  # 021_bleach_cleanser
    13: ( 64,   0,   0),  # 024_bowl
    14: (  0,  64,   0),  # 025_mug
    15: (  0,   0,  64),  # 035_power_drill
    16: ( 64,  64,   0),  # 036_wood_block
    17: ( 64,   0,  64),  # 037_scissors
    18: (  0,  64,  64),  # 040_large_marker
    19: (192,   0,   0),  # 051_large_clamp
    20: (  0, 192,   0),  # 052_extra_large_clamp
    21: (  0,   0, 192),  # 061_foam_brick
}
_MANO_COLOR = (255, 255, 255)


def parse_args():
  parser = argparse.ArgumentParser(
      description='Render hand & object poses in camera views.')
  parser.add_argument('--name',
                      help='Name of the sequence',
                      default=None,
                      type=str)
  parser.add_argument('--device',
                      help='Device for data loader computation',
                      default='cuda:0',
                      type=str)
  args = parser.parse_args()
  return args


class Renderer():
  """Renderer."""

  def __init__(self, name, device='cuda:0'):
    """Constructor.

    Args:
      name: Sequence name.
      device: A torch.device string argument. The specified device is used only
        for certain data loading computations, but not storing the loaded data.
        Currently the loaded data is always stored as numpy arrays on cpu.
    """
    assert device in ('cuda', 'cpu') or device.split(':')[0] == 'cuda'
    self._name = name
    self._device = torch.device(device)

    self._loader = SequenceLoader(self._name,
                                  device=device,
                                  preload=False,
                                  app='renderer')

    # Create pyrender cameras.
    self._cameras = []
    for c in range(self._loader.num_cameras):
      K = self._loader.K[c].cpu().numpy()
      fx = K[0][0].item()
      fy = K[1][1].item()
      cx = K[0][2].item()
      cy = K[1][2].item()
      cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
      self._cameras.append(cam)

    # Create meshes for YCB objects.
    self._mesh_y = []
    for o in range(self._loader.num_ycb):
      obj_file = self._loader.ycb_group_layer.obj_file[o]
      mesh = trimesh.load(obj_file)
      mesh = pyrender.Mesh.from_trimesh(mesh)
      self._mesh_y.append(mesh)

    # Create spheres for MANO joints.
    self._mesh_j = []
    for o in range(self._loader.num_mano):
      mesh = trimesh.creation.uv_sphere(radius=0.005)
      mesh.visual.vertex_colors = [1.0, 0.0, 0.0]
      self._mesh_j.append(mesh)

    self._faces = self._loader.mano_group_layer.f.cpu().numpy()

    w = self._loader.dimensions[0]
    h = self._loader.dimensions[1]
    self._r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

    self._render_dir = [
        os.path.join(os.path.dirname(__file__), "..", "data", "render",
                     self._name, self._loader.serials[c])
        for c in range(self._loader.num_cameras)
    ]
    for d in self._render_dir:
      os.makedirs(d, exist_ok=True)

  def _blend(self, im_real, im_render):
    """Blends the real and rendered images.

    Args:
      im_real: A uint8 numpy array of shape [H, W, 3] containing the real image.
      im_render: A uint8 numpy array of shape [H, W, 3] containing the rendered
        image.
    """
    im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
    im = im.astype(np.uint8)
    return im

  def _render_color_seg(self):
    """Renders and saves color and segmenetation images."""
    print('Rendering color and segmentation')
    for i in range(self._loader.num_frames):
      print('{:03d}/{:03d}'.format(i + 1, self._loader.num_frames))

      self._loader.step()

      for c in range(self._loader.num_cameras):
        # Create pyrender scene.
        scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                               ambient_light=np.array([1.0, 1.0, 1.0]))

        # Add camera.
        scene.add(self._cameras[c], pose=np.eye(4))

        seg_node_map = {}

        pose_y = self._loader.ycb_pose[c]
        vert_m = self._loader.mano_vert[c]

        # Add YCB meshes.
        for o in range(self._loader.num_ycb):
          if np.all(pose_y[o] == 0.0):
            continue
          pose = pose_y[o].copy()
          pose[1] *= -1
          pose[2] *= -1
          node = scene.add(self._mesh_y[o], pose=pose)
          seg_node_map.update({node: _YCB_COLORS[self._loader.ycb_ids[o]]})

        # Add MANO meshes.
        for o in range(self._loader.num_mano):
          if np.all(vert_m[o] == 0.0):
            continue
          vert = vert_m[o].copy()
          vert[:, 1] *= -1
          vert[:, 2] *= -1
          mesh = trimesh.Trimesh(vertices=vert, faces=self._faces)
          mesh1 = pyrender.Mesh.from_trimesh(mesh)
          mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
          mesh2 = pyrender.Mesh.from_trimesh(mesh, wireframe=True)
          mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
          node1 = scene.add(mesh1)
          node2 = scene.add(mesh2)
          seg_node_map.update({node1: _MANO_COLOR})

        color, _ = self._r.render(scene)
        color_seg, _ = self._r.render(scene,
                                      pyrender.RenderFlags.SEG,
                                      seg_node_map=seg_node_map)

        im = self._loader.pcd_rgb[c]
        b_color = self._blend(im, color)
        b_color_seg = self._blend(im, color_seg)

        color_file = self._render_dir[c] + "/color_{:06d}.jpg".format(i)
        seg_file = self._render_dir[c] + "/seg_{:06d}.jpg".format(i)
        cv2.imwrite(color_file, b_color[:, :, ::-1])
        cv2.imwrite(seg_file, b_color_seg[:, :, ::-1])

  def _render_joint(self):
    """Renders and saves hand joint visualizations."""
    print('Rendering joint')
    for i in range(self._loader.num_frames):
      print('{:03d}/{:03d}'.format(i + 1, self._loader.num_frames))

      self._loader.step()

      for c in range(self._loader.num_cameras):
        # Create pyrender scene.
        scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                               ambient_light=np.array([1.0, 1.0, 1.0]))

        # Add camera.
        scene.add(self._cameras[c], pose=np.eye(4))

        joint_3d = self._loader.mano_joint_3d[c]

        # Add MANO joints.
        for o in range(self._loader.num_mano):
          if np.all(joint_3d[o] == -1):
            continue
          j = joint_3d[o].copy()
          j[:, 1] *= -1
          j[:, 2] *= -1
          tfs = np.tile(np.eye(4), (21, 1, 1))
          tfs[:, :3, 3] = j
          mesh = pyrender.Mesh.from_trimesh(self._mesh_j[o], poses=tfs)
          scene.add(mesh)

        color, _ = self._r.render(scene)

        im = self._loader.pcd_rgb[c]
        color = self._blend(im, color)

        color_file = self._render_dir[c] + "/joints_{:06d}.jpg".format(i)
        cv2.imwrite(color_file, color[:, :, ::-1])

  def run(self):
    """Runs the renderer."""
    self._render_color_seg()
    self._render_joint()


if __name__ == '__main__':
  args = parse_args()

  renderer = Renderer(args.name, args.device)
  renderer.run()
