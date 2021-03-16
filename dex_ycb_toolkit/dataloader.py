import torch
import yaml
import glob
import os
import numpy as np
import cv2
import scipy.io as sio

import obj
import vatic

from scipy.spatial.transform import Rotation as Rot

from layers.ycb_group_layer import YCBGroupLayer
from layers.ycb_layer import dcm2rv, rv2dcm, quat2rv
from layers.mano_group_layer import MANOGroupLayer


class DataLoader():

  def __init__(self,
               name,
               device='cuda:0',
               preload=True,
               use_cache=False,
               cap_depth=False,
               load_ycb=False,
               src_ycb='full',
               load_mano=False,
               src_mano='full'):
    """TODO(ywchao): complete docstring.
    Args:
      device: A torch.device string argument. The specified device is used only
        for certain data loading computations, but not storing the loaded data.
        Currently the loaded data is always stored as numpy arrays on cpu.
    """
    assert device in ('cuda', 'cpu') or device.split(':')[0] == 'cuda'
    assert preload or not use_cache
    assert src_ycb in ('pcnn', 'fuse', 'full', 'release')
    assert src_mano in ('kpts', 'full', 'release')
    self._name = name
    self._device = torch.device(device)
    self._preload = preload
    self._use_cache = use_cache
    self._cap_depth = cap_depth
    self._load_ycb = load_ycb
    self._src_ycb = src_ycb
    self._load_mano = load_mano
    self._src_mano = src_mano

    meta_file = "data/" + self._name + "/meta.yml"
    with open(meta_file, 'r') as f:
      meta = yaml.load(f, Loader=yaml.FullLoader)

    self._serials = meta['serials']
    self._num_cameras = len(self._serials)
    self._data_dir = ["data/" + self._name + '/' + s for s in self._serials]
    self._color_prefix = "color_"
    self._depth_prefix = "aligned_depth_to_color_"
    self._num_frames = meta['num_frames']
    self._h = 480
    self._w = 640
    self._depth_bound = 1.0
    self._ycb_ids = meta['ycb_ids']
    self._ycb_grasp_ind = meta['ycb_grasp_ind']
    self._mano_sides = meta['mano_sides']
    if 'pcnn_init' in meta:
      self._pcnn_init = meta['pcnn_init']
    else:
      self._pcnn_init = None
    self._vatic_files_y = [
        "data/" + self._name + '/' + s + "_vatic_ycb/original.txt"
        for s in self._serials
    ]
    self._vatic_files_m = [
        "data/" + self._name + '/' + s + "_vatic_mano/original.txt"
        for s in self._serials
    ]

    def intr_to_K(x):
      return torch.tensor(
          [[x['fx'], 0.0, x['ppx']], [0.0, x['fy'], x['ppy']], [0.0, 0.0, 1.0]],
          dtype=torch.float32,
          device=self._device)

    self._K = []
    for s in self._serials:
      intr_file = "data/calibration/intrinsics/" + s + '_' + str(
          self._w) + 'x' + str(self._h) + ".yml"
      with open(intr_file, 'r') as f:
        intr = yaml.load(f, Loader=yaml.FullLoader)
      K = intr_to_K(intr['color'])
      self._K.append(K)
    self._K_inv = [torch.inverse(k) for k in self._K]

    extr_file = "data/calibration/extrinsics_" + meta[
        'extrinsics'] + "/extrinsics.yml"
    with open(extr_file, 'r') as f:
      extr = yaml.load(f, Loader=yaml.FullLoader)
    T = extr['extrinsics']
    T = {
        s: torch.tensor(T[s], dtype=torch.float32,
                        device=self._device).view(3, 4) for s in T
    }
    self._R = [T[s][:, :3] for s in self._serials]
    self._t = [T[s][:, 3] for s in self._serials]
    self._R_inv = [torch.inverse(r) for r in self._R]
    self._t_inv = [torch.mv(r, -t) for r, t in zip(self._R_inv, self._t)]
    self._master_intrinsics = self._K[[
        i for i, s in enumerate(self._serials) if s == extr['master']
    ][0]].cpu().numpy()
    self._tag_R = T['apriltag'][:, :3]
    self._tag_t = T['apriltag'][:, 3]
    self._tag_R_inv = torch.inverse(self._tag_R)
    self._tag_t_inv = torch.mv(self._tag_R_inv, -self._tag_t)
    self._tag_lim = [-0.00, +1.20, -0.10, +0.70, -0.10, +0.70]

    T_inv = [
        torch.inverse(
            torch.cat((T[s],
                       torch.tensor([[0, 0, 0, 1]],
                                    dtype=torch.float32,
                                    device=self._device))))[:3]
        for s in self._serials
    ]
    self._M = [torch.mm(k, t) for k, t in zip(self._K, T_inv)]

    y, x = torch.meshgrid(torch.arange(self._h), torch.arange(self._w))
    x = x.float()
    y = y.float()
    s = torch.stack((x / (self._w - 1), y / (self._h - 1)), dim=2)
    self._pcd_tex_coord = [s.numpy()] * self._num_cameras

    self._p = []
    ones = torch.ones((self._h, self._w), dtype=torch.float32)
    xy1s = torch.stack((x, y, ones), dim=2).view(self._w * self._h, 3).t()
    xy1s = xy1s.to(self._device)
    for c in range(self._num_cameras):
      p = torch.mm(self._K_inv[c], xy1s)
      self._p.append(p)

    self._rgbd_cache_file = "data/cache/" + self._name + "_rgbd.npz"
    if self._preload:
      print('Preloading point cloud')
      if self._use_cache:
        if not os.path.isfile(self._rgbd_cache_file):
          print('Cache file does not exist: ' + self._rgbd_cache_file)
          exit()
        print('Loading RGB-D from cache: ' + self._rgbd_cache_file)
        cache = np.load(self._rgbd_cache_file)
        self._color = cache['color']
        self._depth = cache['depth']
      else:
        print('Loading RGB-D from frames')
        self._color = []
        self._depth = []
        for c in range(self._num_cameras):
          color = []
          depth = []
          for i in range(self._num_frames):
            rgb, d = self.load_frame_rgbd(c, i)
            color.append(rgb)
            depth.append(d)
          self._color.append(color)
          self._depth.append(depth)
        self._color = np.array(self._color, dtype=np.uint8)
        self._depth = np.array(self._depth, dtype=np.uint16)
      self._pcd_rgb = [x for x in self._color]
      self._pcd_vert = []
      self._pcd_mask = []
      for c in range(self._num_cameras):
        p, m = self.deproject_depth_and_filter_points(self._depth[c], c)
        self._pcd_vert.append(p)
        self._pcd_mask.append(m)
    else:
      print('Loading point cloud online')
      self._pcd_rgb = [
          np.zeros((self._h, self._w, 3), dtype=np.uint8)
          for _ in range(self._num_cameras)
      ]
      self._pcd_vert = [
          np.zeros((self._h, self._w, 3), dtype=np.float32)
          for _ in range(self._num_cameras)
      ]
      self._pcd_mask = [
          np.zeros((self._h, self._w), dtype=np.bool)
          for _ in range(self._num_cameras)
      ]

    self._ycb_group_layer = YCBGroupLayer(self._ycb_ids).to(self._device)

    if self._load_ycb:
      self._ycb_count = self._ycb_group_layer.count
      self._ycb_material = self._ycb_group_layer.material
      self._ycb_tex_coords = self._ycb_group_layer.tex_coords

      s = np.cumsum([0] + self._ycb_group_layer.count[:-1])
      e = np.cumsum(self._ycb_group_layer.count)
      self._ycb_seg = list(zip(s, e))

      # TODO(ywchao): self._ycb_mask
      if self._src_ycb == 'pcnn':
        self._pcnn_dir = [d + "_posecnn_results" for d in self._data_dir]
        self._pcnn_cache_file = "data/cache/" + self._name + "_pcnn.npz"
        if self._preload:
          print('Preloading PoseCNN pose')
          if self._use_cache:
            if not os.path.isfile(self._pcnn_cache_file):
              print('Cache file does not exist: ' + self._pcnn_cache_file)
              exit()
            print('Loading PoseCNN pose from cache: ' + self._pcnn_cache_file)
            cache = np.load(self._pcnn_cache_file)
            self._ycb_pcnn = cache['ycb_pcnn']
          else:
            print('Loading PoseCNN pose from PoseCNN output')
            self._ycb_pcnn = []
            for c in range(self._num_cameras):
              ycb_pcnn = []
              for i in range(self._num_frames):
                _, pose = self.load_frame_ycb_pcnn(c, i)
                ycb_pcnn.append(pose)
              self._ycb_pcnn.append(ycb_pcnn)
            self._ycb_pcnn = np.array(self._ycb_pcnn, dtype=np.float32)
          self._ycb_vert = []
          self._ycb_norm = []
          for c in range(self._num_cameras):
            ycb_vert = [
                np.zeros((self._num_frames, n, 3), dtype=np.float32)
                for n in self._ycb_count
            ]
            ycb_norm = [
                np.zeros((self._num_frames, n, 3), dtype=np.float32)
                for n in self._ycb_count
            ]
            i = np.any(self._ycb_pcnn[c] != [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       axis=2)
            pose = self._ycb_pcnn[c].reshape(-1, 7)
            v, n = self.transform_ycb(pose, c=c)
            for o in range(self._ycb_group_layer.num_obj):
              io = i[:, o]
              ycb_vert[o][io] = v[io, self._ycb_seg[o][0]:self._ycb_seg[o][1]]
              ycb_norm[o][io] = n[io, self._ycb_seg[o][0]:self._ycb_seg[o][1]]
            self._ycb_vert.append(ycb_vert)
            self._ycb_norm.append(ycb_norm)
        else:
          print('Loading PoseCNN pose online')
          self._ycb_vert = [[
              np.zeros((n, 3), dtype=np.float32) for n in self._ycb_count
          ] for _ in range(self._num_cameras)]
          self._ycb_norm = [[
              np.zeros((n, 3), dtype=np.float32) for n in self._ycb_count
          ] for _ in range(self._num_cameras)]
          self._pcnn_labels = [None for _ in range(self._num_cameras)]
          self._pcnn_rois = [None for _ in range(self._num_cameras)]
          self._pcnn_poses = [None for _ in range(self._num_cameras)]

      if self._src_ycb in ('fuse', 'full', 'release'):
        if self._src_ycb == 'fuse':
          ycb_file = "data/" + self._name + "/solve_ycb.npz"
        if self._src_ycb == 'full':
          ycb_file = "data/" + self._name + "/solve_joint.npz"
        if self._src_ycb == 'release':
          ycb_file = "data/" + self._name + "/pose.npz"
        data = np.load(ycb_file)
        self._ycb_pose = data['pose_y']
        i = np.any(self._ycb_pose != [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                   axis=2)
        pose = self._ycb_pose.reshape(-1, 7)
        v, n = self.transform_ycb(pose)
        self._ycb_vert = [
            np.zeros((self._num_frames, n, 3), dtype=np.float32)
            for n in self._ycb_count
        ]
        self._ycb_norm = [
            np.zeros((self._num_frames, n, 3), dtype=np.float32)
            for n in self._ycb_count
        ]
        for o in range(self._ycb_group_layer.num_obj):
          io = i[:, o]
          self._ycb_vert[o][io] = v[io, self._ycb_seg[o][0]:self._ycb_seg[o][1]]
          self._ycb_norm[o][io] = n[io, self._ycb_seg[o][0]:self._ycb_seg[o][1]]
        if self._src_ycb in ('fuse', 'full'):
          self._ycb_mask = np.swapaxes(data['labs_y'], 0, 1)

    mano_betas = []
    for m in meta['mano_calib']:
      mano_calib_file = "data/calibration/mano_" + m + "/mano.yml"
      with open(mano_calib_file, 'r') as f:
        mano_calib = yaml.load(f, Loader=yaml.FullLoader)
      betas = np.array(mano_calib['betas'], dtype=np.float32)
      mano_betas.append(betas)

    self._mano_group_layer = MANOGroupLayer(self._mano_sides,
                                            mano_betas).to(self._device)

    if self._load_mano:
      if self._src_mano == 'kpts':
        mano_file = "data/" + self._name + "/solve_mano.npz"
      if self._src_mano == 'full':
        mano_file = "data/" + self._name + "/solve_joint.npz"
      if self._src_mano == 'release':
        mano_file = "data/" + self._name + "/pose.npz"
      data = np.load(mano_file)
      self._mano_pose = data['pose_m']
      i = np.any(self._mano_pose != 0.0, axis=2)
      pose = torch.from_numpy(self._mano_pose).to(self._device)
      pose = pose.view(-1, self._mano_group_layer.num_obj * 51)
      verts, _ = self._mano_group_layer(pose)
      # Numpy array is faster than PyTorch Tensor here.
      verts = verts.cpu().numpy()
      f = self._mano_group_layer.f.cpu().numpy()
      v = verts[:, f.ravel()]
      n = np.cross(v[:, 1::3, :] - v[:, 0::3, :], v[:, 2::3, :] - v[:, 1::3, :])
      n = np.repeat(n, 3, axis=1)
      l = verts[:, f[:, [0, 1, 1, 2, 2, 0]].ravel(), :]
      self._mano_vert = [
          np.zeros((self._num_frames, 4614, 3), dtype=np.float32)
          for _ in range(self._mano_group_layer.num_obj)
      ]
      self._mano_norm = [
          np.zeros((self._num_frames, 4614, 3), dtype=np.float32)
          for _ in range(self._mano_group_layer.num_obj)
      ]
      self._mano_line = [
          np.zeros((self._num_frames, 9228, 3), dtype=np.float32)
          for _ in range(self._mano_group_layer.num_obj)
      ]
      for o in range(self._mano_group_layer.num_obj):
        io = i[:, o]
        self._mano_vert[o][io] = v[io, 4614 * o:4614 * (o + 1), :]
        self._mano_norm[o][io] = n[io, 4614 * o:4614 * (o + 1), :]
        self._mano_line[o][io] = l[io, 9228 * o:9228 * (o + 1), :]
      if self._src_mano == 'full':
        self._mano_mask = np.swapaxes(data['labs_m'], 0, 1)

    self._frame = -1

  def save_cache(self):
    print('Saving RGB-D to cache: ' + self._rgbd_cache_file)
    np.savez(self._rgbd_cache_file, color=self._color, depth=self._depth)
    if self._load_ycb and self._src_ycb == 'pcnn':
      print('Saving PoseCNN pose to cache: ' + self._pcnn_cache_file)
      np.savez(self._pcnn_cache_file, ycb_pcnn=self._ycb_pcnn)

  def load_frame_rgbd(self, c, i):
    color_file = self._data_dir[
        c] + '/' + self._color_prefix + "{:06d}.jpg".format(i)
    color = cv2.imread(color_file)
    color = color[:, :, ::-1]
    depth_file = self._data_dir[
        c] + '/' + self._depth_prefix + "{:06d}.png".format(i)
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    return color, depth

  def deproject_depth_and_filter_points(self, d, c):
    nd = d.ndim
    d = d.astype(np.float32) / 1000
    d = torch.from_numpy(d).to(self._device)
    p = torch.mul(
        d.view(1, -1, self._w * self._h).expand(3, -1, -1),
        self._p[c].unsqueeze(1))
    p = torch.addmm(self._t[c].unsqueeze(1), self._R[c], p.view(3, -1))
    p_tag = torch.addmm(self._tag_t_inv.unsqueeze(1), self._tag_R_inv, p)
    mx1 = p_tag[0, :] > self._tag_lim[0]
    mx2 = p_tag[0, :] < self._tag_lim[1]
    my1 = p_tag[1, :] > self._tag_lim[2]
    my2 = p_tag[1, :] < self._tag_lim[3]
    mz1 = p_tag[2, :] > self._tag_lim[4]
    mz2 = p_tag[2, :] < self._tag_lim[5]
    m = mx1 & mx2 & my1 & my2 & mz1 & mz2
    if self._cap_depth:
      m &= d.view(-1) < self._depth_bound
    p = p.t().view(-1, self._h, self._w, 3)
    m = m.view(-1, self._h, self._w)
    if nd == 2:
      p = p.squeeze(0)
      m = m.squeeze(0)
    p = p.cpu().numpy()
    m = m.cpu().numpy()
    return p, m

  def load_frame_ycb_pcnn(self, c, i):
    pcnn_file = self._pcnn_dir[
        c] + '/' + self._color_prefix + "{:06d}.jpg.mat".format(i)
    pcnn = sio.loadmat(pcnn_file)
    # Handle the missing class (051_large_clamp) in PCNN.
    pcnn['labels'][pcnn['labels'] > 18] += 1
    if len(pcnn['rois']) > 0:
      pcnn['rois'][:, 1][pcnn['rois'][:, 1] > 18] += 1
    pose = np.zeros((self._ycb_group_layer.num_obj, 7), dtype=np.float32)
    pose[:, 3] = 1.0
    if len(pcnn['rois']) > 0:
      for o, i in enumerate(self._ycb_ids):
        ind = np.where(pcnn['rois'][:, 1] == i)[0]
        if len(ind) > 0:
          j = np.argmax(pcnn['rois'][ind, 6])
          pose[o] = pcnn['poses'][ind[j], [1, 2, 3, 0, 4, 5, 6]]
    return pcnn, pose

  def transform_ycb(self,
                    pose,
                    c=None,
                    camera_to_world=True,
                    run_ycb_group_layer=True,
                    return_trans_mat=False):
    if pose.shape[1] == 7:
      q = pose[:, :4]
      t = pose[:, 4:]
      R = Rot.from_quat(q).as_dcm().astype(np.float32)
      R = torch.from_numpy(R).to(self._device)
      t = torch.from_numpy(t).to(self._device)
    if pose.shape[1] == 6:
      r = pose[:, :3]
      t = pose[:, 3:]
      r = torch.from_numpy(r).to(self._device)
      t = torch.from_numpy(t).to(self._device)
      R = rv2dcm(r)
    if c is not None:
      if camera_to_world:
        R_c = self._R[c]
        t_c = self._t[c]
      else:
        R_c = self._R_inv[c]
        t_c = self._t_inv[c]
      R = torch.bmm(R_c.expand(R.size(0), -1, -1), R)
      t = torch.addmm(t_c, t, R_c.t())
    if run_ycb_group_layer or not return_trans_mat:
      r = dcm2rv(R)
      p = torch.cat([r, t], dim=1)
    else:
      p = torch.cat([R, t.unsqueeze(2)], dim=2)
      p = torch.cat([
          p,
          torch.tensor([[[0, 0, 0, 1]]] * R.size(0),
                       dtype=torch.float32,
                       device=self._device)
      ],
                    dim=1)
    if run_ycb_group_layer:
      p = p.view(-1, self._ycb_group_layer.num_obj * 6)
      v, n = self._ycb_group_layer(p)
      v = v[:, self._ycb_group_layer.f.view(-1)]
      n = n[:, self._ycb_group_layer.f.view(-1)]
      v = v.cpu().numpy()
      n = n.cpu().numpy()
      return v, n
    else:
      p = p.cpu().numpy()
      return p

  @property
  def serials(self):
    return self._serials

  @property
  def num_cameras(self):
    return self._num_cameras

  @property
  def num_frames(self):
    return self._num_frames

  @property
  def dimensions(self):
    return self._w, self._h

  @property
  def ycb_ids(self):
    return self._ycb_ids

  @property
  def ycb_grasp_ind(self):
    return self._ycb_grasp_ind

  @property
  def mano_sides(self):
    return self._mano_sides

  @property
  def pcnn_init(self):
    return self._pcnn_init

  @property
  def K(self):
    return self._K

  @property
  def K_inv(self):
    return self._K_inv

  @property
  def master_intrinsics(self):
    return self._master_intrinsics

  @property
  def M(self):
    return self._M

  @property
  def frame(self):
    return self._frame

  def step(self):
    self._frame = (self._frame + 1) % self._num_frames
    if not self._preload:
      self.update_pcd()
    if not self._preload and self._src_ycb == 'pcnn':
      self.update_ycb_pcnn()

  def update_pcd(self):
    for c in range(self._num_cameras):
      rgb, d = self.load_frame_rgbd(c, self._frame)
      p, m = self.deproject_depth_and_filter_points(d, c)
      self._pcd_rgb[c][:] = rgb
      self._pcd_vert[c][:] = p
      self._pcd_mask[c][:] = m

  def update_ycb_pcnn(self):
    for c in range(self._num_cameras):
      pcnn, pose = self.load_frame_ycb_pcnn(c, self._frame)
      i = np.any(pose != [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=1)
      v, n = self.transform_ycb(pose, c=c)
      for o in range(self._ycb_group_layer.num_obj):
        if i[o]:
          self._ycb_vert[c][o][:] = v[:, self._ycb_seg[o][0]:self.
                                      _ycb_seg[o][1]]
          self._ycb_norm[c][o][:] = n[:, self._ycb_seg[o][0]:self.
                                      _ycb_seg[o][1]]
        else:
          self._ycb_vert[c][o][:] = 0
          self._ycb_vert[c][o][:] = 0
      rois = pcnn['rois']
      poses = pcnn['poses']
      if len(rois) == 0 or np.all(rois[:, 1] == 0):
        rois = np.zeros((0, 7), dtype=np.float32)
        poses = np.zeros((0, 6), dtype=np.float32)
      else:
        poses[:, :4] = poses[:, [1, 2, 3, 0]]
        poses = self.transform_ycb(poses, c=c, run_ycb_group_layer=False)
      self._pcnn_labels[c] = pcnn['labels']
      self._pcnn_rois[c] = rois
      self._pcnn_poses[c] = poses

  @property
  def pcd_rgb(self):
    if self._preload:
      return [x[self._frame] for x in self._pcd_rgb]
    else:
      return self._pcd_rgb

  @property
  def pcd_vert(self):
    if self._preload:
      return [x[self._frame] for x in self._pcd_vert]
    else:
      return self._pcd_vert

  @property
  def pcd_tex_coord(self):
    return self._pcd_tex_coord

  @property
  def pcd_mask(self):
    if self._preload:
      return [x[self._frame] for x in self._pcd_mask]
    else:
      return self._pcd_mask

  @property
  def load_ycb(self):
    return self._load_ycb

  @property
  def ycb_group_layer(self):
    return self._ycb_group_layer

  @property
  def num_ycb(self):
    return self._ycb_group_layer.num_obj

  @property
  def ycb_count(self):
    return self._ycb_count

  @property
  def ycb_material(self):
    return self._ycb_material

  @property
  def ycb_pose(self):
    if self._src_ycb == 'pcnn':
      return None
    if self._src_ycb in ('fuse', 'full', 'release'):
      return self._ycb_pose

  @property
  def ycb_vert(self):
    if self._src_ycb == 'pcnn':
      if self._preload:
        return [[y[self._frame] for y in x] for x in self._ycb_vert]
      else:
        return self._ycb_vert
    if self._src_ycb in ('fuse', 'full', 'release'):
      return [x[self._frame] for x in self._ycb_vert]

  @property
  def ycb_norm(self):
    if self._src_ycb == 'pcnn':
      if self._preload:
        return [[y[self._frame] for y in x] for x in self._ycb_norm]
      else:
        return self._ycb_norm
    if self._src_ycb in ('fuse', 'full', 'release'):
      return [x[self._frame] for x in self._ycb_norm]

  @property
  def ycb_tex_coords(self):
    return self._ycb_tex_coords

  @property
  def ycb_mask(self):
    if self._src_ycb in ('pcnn', 'release'):
      return None
    if self._src_ycb in ('fuse', 'full'):
      return [x[self._frame] for x in self._ycb_mask]

  @property
  def pcnn_labels(self):
    return self._pcnn_labels

  @property
  def pcnn_rois(self):
    return self._pcnn_rois

  @property
  def pcnn_poses(self):
    return self._pcnn_poses

  @property
  def load_mano(self):
    return self._load_mano

  @property
  def mano_group_layer(self):
    return self._mano_group_layer

  @property
  def num_mano(self):
    return self._mano_group_layer.num_obj

  @property
  def mano_pose(self):
    return self._mano_pose

  @property
  def mano_vert(self):
    return [x[self._frame] for x in self._mano_vert]

  @property
  def mano_norm(self):
    return [x[self._frame] for x in self._mano_norm]

  @property
  def mano_line(self):
    return [x[self._frame] for x in self._mano_line]

  @property
  def mano_mask(self):
    if self._src_mano in ('kpts', 'release'):
      return None
    if self._src_mano == 'full':
      return [x[self._frame] for x in self._mano_mask]

  def load_vatic_kpts_ycb(self, labels, radius=5):
    return self.load_vatic_kpts(self._vatic_files_y, labels, radius)

  def load_vatic_kpts_mano(self, labels, radius=5):
    return self.load_vatic_kpts(self._vatic_files_m, labels, radius)

  def load_vatic_kpts(self, vatic_files, labels, radius=5):
    kpts = []
    for f in vatic_files:
      if os.path.isfile(f):
        out = vatic.read_vatic_output(f,
                                      labels,
                                      radius,
                                      run_assert=True,
                                      num_frames=self._num_frames)
      else:
        out = -1 * np.ones((self._num_frames, len(labels), 2), dtype=np.int32)
      kpts.append(out)
    kpts = np.stack(kpts, axis=1)
    return kpts

  def transform_tag(self, e, t):
    R = Rot.from_euler('ZYX', e).as_dcm().astype(np.float32)
    R = torch.from_numpy(R).to(self._device)
    t = torch.from_numpy(t).to(self._device)
    R = torch.bmm(self._tag_R.expand(R.size(0), -1, -1), R)
    t = torch.addmm(self._tag_t, t, self._tag_R.t())
    r = dcm2rv(R)
    r = r.cpu().numpy()
    t = t.cpu().numpy()
    return r, t
