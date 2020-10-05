import numpy as np
import pyrender
import os
import trimesh
import cv2
import matplotlib.pyplot as plt

from lib.factory import get_dataset

name = 's0_train'
dataset = get_dataset(name)

idx = 0

sample = dataset[idx]


def create_scene(sample):
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

  # Load YCB meshes.
  mesh_y = []
  for i in sample['ycb_ids']:
    obj_file = dataset.obj_file[i]
    mesh = trimesh.load(obj_file)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_y.append(mesh)

  # Load YCB pose.
  label = np.load(sample['label_file'])
  pose_y = label['pose_y']

  # Add YCB meshes.
  for o in range(len(sample['ycb_ids'])):
    if np.all(pose_y[o] == 0.0):
      continue
    pose = np.vstack((pose_y[o], np.array([[0, 0, 0, 1]], dtype=np.float32)))
    pose[1] *= -1
    pose[2] *= -1
    node = scene.add(mesh_y[o], pose=pose)

  return scene


print('Visualize pose in camera view using pyrender renderer')

scene = create_scene(sample)

r = pyrender.OffscreenRenderer(viewport_width=dataset.w,
                               viewport_height=dataset.h)

im_render, _ = r.render(scene)

im_real = cv2.imread(sample['color_file'])
im_real = im_real[:, :, ::-1]

im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
im = im.astype(np.uint8)

plt.ion()
plt.imshow(im)
plt.tight_layout()
plt.pause(0.0001)

print('Visualize pose using pyrender 3D viewer')

scene = create_scene(sample)

pyrender.Viewer(scene)
