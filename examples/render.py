import argparse
import torch
import pyrender
import trimesh
import numpy as np
import os
import time
import cv2

import _init_paths
import dataloader

_YCB_COLORS = [
    (  0,   0,   0),  # __background__
    (255,   0,   0),  # 002_master_chef_can
    (  0, 255,   0),  # 003_cracker_box
    (  0,   0, 255),  # 004_sugar_box
    (255, 255,   0),  # 005_tomato_soup_can
    (255,   0, 255),  # 006_mustard_bottle
    (  0, 255, 255),  # 007_tuna_fish_can
    (128,   0,   0),  # 008_pudding_box
    (  0, 128,   0),  # 009_gelatin_box
    (  0,   0, 128),  # 010_potted_meat_can
    (128, 128,   0),  # 011_banana
    (128,   0, 128),  # 019_pitcher_base
    (  0, 128, 128),  # 021_bleach_cleanser
    ( 64,   0,   0),  # 024_bowl
    (  0,  64,   0),  # 025_mug
    (  0,   0,  64),  # 035_power_drill
    ( 64,  64,   0),  # 036_wood_block
    ( 64,   0,  64),  # 037_scissors
    (  0,  64,  64),  # 040_large_marker
    (192,   0,   0),  # 051_large_clamp
    (  0, 192,   0),  # 052_extra_large_clamp
    (  0,   0, 192),  # 061_foam_brick
]
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
  parser.add_argument('--load-ycb', action='store_true', default=False)
  parser.add_argument('--src-ycb',
                      help='Source of the YCB pose',
                      default='full',
                      type=str,
                      choices=['fuse', 'full', 'release'])
  parser.add_argument('--load-mano', action='store_true', default=False)
  parser.add_argument('--src-mano',
                      help='Source of the MANO pose',
                      default='full',
                      type=str,
                      choices=['kpts', 'full', 'release'])
  args = parser.parse_args()
  return args


class Renderer():

  def __init__(self, name, device, load_ycb, src_ycb, load_mano, src_mano):
    assert device in ('cuda', 'cpu') or device.split(':')[0] == 'cuda'
    self._name = name
    self._device = torch.device(device)
    self._load_ycb = load_ycb
    self._src_ycb = src_ycb
    self._load_mano = load_mano
    self._src_mano = src_mano

    self._loader = dataloader.DataLoader(args.name,
                                         device=device,
                                         preload=False,
                                         load_ycb=self._load_ycb,
                                         src_ycb=self._src_ycb,
                                         load_mano=self._load_mano,
                                         src_mano=self._src_mano)

    self._cameras = []
    for c in range(self._loader.num_cameras):
      K = self._loader.K[c].cpu().numpy()
      fx = K[0][0].item()
      fy = K[1][1].item()
      cx = K[0][2].item()
      cy = K[1][2].item()
      cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
      self._cameras.append(cam)

    self._mesh_y = []
    for o in range(self._loader.num_ycb):
      obj_file = self._loader.ycb_group_layer.obj_file[o]
      mesh = trimesh.load(obj_file)
      mesh = pyrender.Mesh.from_trimesh(mesh)
      self._mesh_y.append(mesh)

    self._mesh_j = []
    for o in range(self._loader.num_mano):
      mesh = trimesh.creation.uv_sphere(radius=0.005)
      mesh.visual.vertex_colors = [1.0, 0.0, 0.0]
      self._mesh_j.append(mesh)

    # TODO(ywchao): remove skip_y and skip_m and change the values of pose/j3d/j2d to 0 or -1?

    if self._load_ycb:
      self._skip_y = np.all(
          self._loader.ycb_pose == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=2)
      self._pose_y = []
      for c in range(self._loader.num_cameras):
        p = self._loader.ycb_pose.reshape(-1, 7)
        p = self._loader.transform_ycb(p,
                                       c=c,
                                       camera_to_world=False,
                                       run_ycb_group_layer=False,
                                       return_trans_mat=True)
        p = p.reshape(-1, self._loader.num_ycb, 4, 4)
        self._pose_y.append(p)

    if self._load_mano:
      self._skip_m = np.all(self._loader.mano_pose == 0.0, axis=2)
      self._pose_m = []
      self._vert_m = []
      self._j3d_m = []
      self._j2d_m = []
      root_trans = self._loader.mano_group_layer.root_trans.cpu().numpy()
      root_trans = np.tile(root_trans, (len(self._loader.mano_pose), 1))
      for c in range(self._loader.num_cameras):
        p = self._loader.mano_pose.reshape(-1, 51).copy()
        r = p[:, np.r_[0:3, 48:51]]
        r[:, 3:] += root_trans
        r = self._loader.transform_ycb(r,
                                       c=c,
                                       camera_to_world=False,
                                       run_ycb_group_layer=False)
        r[:, 3:] -= root_trans
        p[:, np.r_[0:3, 48:51]] = r
        p = p.reshape(-1, self._loader.num_mano, 51)
        self._pose_m.append(p)
        p = torch.from_numpy(p).to(self._device)
        p = p.view(-1, self._loader.num_mano * 51)
        v, j = self._loader.mano_group_layer(p)
        v = v.view(-1, self._loader.num_mano, 778, 3)
        v = v.cpu().numpy()
        self._vert_m.append(v)
        k = j.view(-1, 3)
        k = torch.mm(k, self._loader.K[c].t())
        k = k[:, :2] / k[:, [2]]
        k = k.view(-1, self._loader.num_mano, 21, 2)
        k = k.cpu().numpy()
        self._j2d_m.append(k)
        j = j.view(-1, self._loader.num_mano, 21, 3)
        j = j.cpu().numpy()
        self._j3d_m.append(j)
      self._faces = self._loader.mano_group_layer.f.cpu().numpy()

    w = self._loader.dimensions[0]
    h = self._loader.dimensions[1]
    self._r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

    self._render_dir = [
        "data/" + self._name + '/' + self._loader.serials[c] + "_render"
        for c in range(self._loader.num_cameras)
    ]
    for d in self._render_dir:
      os.makedirs(d, exist_ok=True)

  def blend(self, im_real, im_render):
    im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
    im = im.astype(np.uint8)
    return im

  def run(self):
    for i in range(self._loader.num_frames):
      print('{:03d}/{:03d}'.format(i + 1, self._loader.num_frames))
      s = time.time()

      self._loader.step()

      for c in range(self._loader.num_cameras):
        scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                               ambient_light=np.array([1.0, 1.0, 1.0]))

        scene.add(self._cameras[c], pose=np.eye(4))

        seg_node_map = {}

        if self._load_ycb:
          for o in range(self._loader.num_ycb):
            if self._skip_y[i][o]:
              continue
            pose = self._pose_y[c][i][o].copy()
            pose[1] *= -1
            pose[2] *= -1
            node = scene.add(self._mesh_y[o], pose=pose)
            seg_node_map.update({node: _YCB_COLORS[self._loader.ycb_ids[o]]})

        if self._load_mano:
          for o in range(self._loader.num_mano):
            if self._skip_m[i][o]:
              continue
            vert = self._vert_m[c][i][o].copy()
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
        # color, _ = self._r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        # color_seg, _ = self._r.render(scene,
        #                               pyrender.RenderFlags.SEG | pyrender.RenderFlags.SKIP_CULL_FACES,
        #                               seg_node_map=seg_node_map)

        im = self._loader.pcd_rgb[c]
        b_color = self.blend(im, color)
        b_color_seg = self.blend(im, color_seg)

        color_file = self._render_dir[c] + "/color_{:06d}.jpg".format(i)
        seg_file = self._render_dir[c] + "/seg_{:06d}.jpg".format(i)
        cv2.imwrite(color_file, b_color[:, :, ::-1])
        cv2.imwrite(seg_file, b_color_seg[:, :, ::-1])

        seg = np.zeros_like(color_seg[:, :, 0])
        for x in self._loader.ycb_ids:
          seg[np.all(color_seg == _YCB_COLORS[x], axis=2)] = x
        seg[np.all(color_seg == _MANO_COLOR, axis=2)] = 255

        pose_y = self._pose_y[c][i][:, :3].copy()
        for o in range(self._loader.num_ycb):
          if self._skip_y[i][o]:
            pose_y[o][:] = 0.0

        pose_m = self._pose_m[c][i].copy()
        for o in range(self._loader.num_mano):
          if self._skip_m[i][o]:
            assert np.all(pose_m[o][3:48] == 0.0)
            pose_m[o][:] = 0.0

        joint_3d = self._j3d_m[c][i]
        joint_2d = self._j2d_m[c][i]
        for o in range(self._loader.num_mano):
          if self._skip_m[i][o]:
            joint_3d[o] = -1
            joint_2d[o] = -1

        labels_file = self._render_dir[c] + "/labels_{:06d}.npz".format(i)
        np.savez_compressed(labels_file, seg=seg, pose_y=pose_y, pose_m=pose_m, joint_3d=joint_3d, joint_2d=joint_2d)

      e = time.time()
      print('time: {:6.2f}'.format(e - s))

    for i in range(self._loader.num_frames):
      print('{:03d}/{:03d}'.format(i + 1, self._loader.num_frames))
      s = time.time()

      self._loader.step()

      for c in range(self._loader.num_cameras):
        scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                               ambient_light=np.array([1.0, 1.0, 1.0]))

        scene.add(self._cameras[c], pose=np.eye(4))

        if self._load_mano:
          for o in range(self._loader.num_mano):
            if self._skip_m[i][o]:
              continue
            j = self._j3d_m[c][i][o].copy()
            j[:, 1] *= -1
            j[:, 2] *= -1
            tfs = np.tile(np.eye(4), (21, 1, 1))
            tfs[:, :3, 3] = j
            mesh = pyrender.Mesh.from_trimesh(self._mesh_j[o], poses=tfs)
            scene.add(mesh)

        color, _ = self._r.render(scene)

        im = self._loader.pcd_rgb[c]
        color = self.blend(im, color)

        color_file = self._render_dir[c] + "/joints_{:06d}_1.jpg".format(i)
        cv2.imwrite(color_file, color[:, :, ::-1])

      e = time.time()
      print('time: {:6.2f}'.format(e - s))

if __name__ == '__main__':
  args = parse_args()

  renderer = Renderer(args.name, args.device, args.load_ycb, args.src_ycb,
                      args.load_mano, args.src_mano)
  renderer.run()
