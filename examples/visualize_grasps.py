# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of visualizing pre-generating grasps for each YCB object."""

import os
import json
import trimesh
import pyrender
import numpy as np
import copy

from dex_ycb_toolkit.factory import get_dataset


def main():
  dataset = get_dataset('s0_train')

  # Load pre-generated grasps for YCB objects.
  ycb_grasp_file = os.path.join(os.path.dirname(__file__), "..", "assets",
                                "ycb_farthest_100_grasps.json")
  with open(ycb_grasp_file, 'r') as f:
    ycb_grasps = json.load(f)

  # Load simplified panda gripper mesh.
  gripper_mesh_file = os.path.join(os.path.dirname(__file__), "..", "assets",
                                   "panda_tubes.obj")
  gripper_mesh = trimesh.load(gripper_mesh_file)

  gripper_material = pyrender.MetallicRoughnessMaterial(
      alphaMode="BLEND",
      doubleSided=True,
      baseColorFactor=(0.00, 1.00, 0.04, 1.00),
      metallicFactor=0.0)

  # Visualize pre-generated grasps for each YCB object.
  for ycb_id, name in dataset.ycb_classes.items():
    if name not in ycb_grasps.keys():
      print('{} does not have pre-generated grasps: skip.'.format(name))
      continue

    scene = pyrender.Scene(ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))

    obj_mesh = trimesh.load(dataset.obj_file[ycb_id])
    scene.add(pyrender.Mesh.from_trimesh(obj_mesh))

    for grasp in ycb_grasps[name]:
      grasp_mesh = copy.deepcopy(gripper_mesh).apply_transform(grasp),
      scene.add(pyrender.Mesh.from_trimesh(grasp_mesh, gripper_material))

    print('Visualizing pre-generated grasps of {}'.format(name))
    print('Close the window to visualize next object.')

    pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == '__main__':
  main()
