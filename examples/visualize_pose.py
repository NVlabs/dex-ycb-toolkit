# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of visualizing object and hand pose of one image sample."""

import numpy as np
import pyrender
import trimesh
import torch
import cv2
import matplotlib.pyplot as plt

from manopth.manolayer import ManoLayer

from dex_ycb_toolkit.factory import get_dataset


def create_scene(sample, obj_file):
  """Creates the pyrender scene of an image sample.

  Args:
    sample: A dictionary holding an image sample.
    obj_file: A dictionary holding the paths to YCB OBJ files.

  Returns:
    A pyrender scene object.
  """
  # Create pyrender scene.
  scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                         ambient_light=np.array([1.0, 1.0, 1.0]))

  # Add camera.
  fx = sample['intrinsics']['fx']
  fy = sample['intrinsics']['fy']
  cx = sample['intrinsics']['ppx']
  cy = sample['intrinsics']['ppy']
  cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
  scene.add(cam, pose=np.eye(4))

  # Load poses.
  label = np.load(sample['label_file'])
  pose_y = label['pose_y']
  pose_m = label['pose_m']

  # Load YCB meshes.
  mesh_y = []
  for i in sample['ycb_ids']:
    mesh = trimesh.load(obj_file[i])
    mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_y.append(mesh)

  # Add YCB meshes.
  for o in range(len(pose_y)):
    if np.all(pose_y[o] == 0.0):
      continue
    pose = np.vstack((pose_y[o], np.array([[0, 0, 0, 1]], dtype=np.float32)))
    pose[1] *= -1
    pose[2] *= -1
    node = scene.add(mesh_y[o], pose=pose)

  # Load MANO layer.
  mano_layer = ManoLayer(flat_hand_mean=False,
                         ncomps=45,
                         side=sample['mano_side'],
                         mano_root='manopth/mano/models',
                         use_pca=True)
  faces = mano_layer.th_faces.numpy()
  betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)

  # Add MANO meshes.
  if not np.all(pose_m == 0.0):
    pose = torch.from_numpy(pose_m)
    vert, _ = mano_layer(pose[:, 0:48], betas, pose[:, 48:51])
    vert /= 1000
    vert = vert.view(778, 3)
    vert = vert.numpy()
    vert[:, 1] *= -1
    vert[:, 2] *= -1
    mesh = trimesh.Trimesh(vertices=vert, faces=faces)
    mesh1 = pyrender.Mesh.from_trimesh(mesh)
    mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
    mesh2 = pyrender.Mesh.from_trimesh(mesh, wireframe=True)
    mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
    node1 = scene.add(mesh1)
    node2 = scene.add(mesh2)

  return scene


def main():
  name = 's0_train'
  dataset = get_dataset(name)

  idx = 70

  sample = dataset[idx]

  scene_r = create_scene(sample, dataset.obj_file)
  scene_v = create_scene(sample, dataset.obj_file)

  print('Visualizing pose in camera view using pyrender renderer')

  r = pyrender.OffscreenRenderer(viewport_width=dataset.w,
                                 viewport_height=dataset.h)

  im_render, _ = r.render(scene_r)

  im_real = cv2.imread(sample['color_file'])
  im_real = im_real[:, :, ::-1]

  im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
  im = im.astype(np.uint8)

  print('Close the window to continue.')

  plt.imshow(im)
  plt.tight_layout()
  plt.show()

  print('Visualizing pose using pyrender 3D viewer')

  pyrender.Viewer(scene_v)


if __name__ == '__main__':
  main()
