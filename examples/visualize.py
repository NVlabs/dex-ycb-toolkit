import numpy as np
import pyrender
import trimesh
import torch
import cv2
import matplotlib.pyplot as plt

from dex_ycb_toolkit.factory import get_dataset
from dex_ycb_toolkit.layers.mano_group_layer import MANOGroupLayer


def create_scene(sample, obj_file):
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

  # Load MANO group layer.
  mano_sides = [sample['mano_side']]
  mano_betas = [np.array(sample['mano_betas'], dtype=np.float32)]
  mano_group_layer = MANOGroupLayer(mano_sides, mano_betas)
  faces = mano_group_layer.f.numpy()

  # Add MANO meshes.
  if not np.all(pose_m == 0.0):
    pose_m = torch.from_numpy(pose_m)
    vert, _ = mano_group_layer(pose_m)
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

  print('Visualizing pose using pyrender 3D viewer')

  scene = create_scene(sample, dataset.obj_file)

  pyrender.Viewer(scene, run_in_thread=True)

  print('Visualizing pose in camera view using pyrender renderer')

  scene = create_scene(sample, dataset.obj_file)

  r = pyrender.OffscreenRenderer(viewport_width=dataset.w,
                                 viewport_height=dataset.h)

  im_render, _ = r.render(scene)

  im_real = cv2.imread(sample['color_file'])
  im_real = im_real[:, :, ::-1]

  im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
  im = im.astype(np.uint8)

  plt.imshow(im)
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  main()
