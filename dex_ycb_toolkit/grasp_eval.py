# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Grasp evaluator."""

import os
import sys
import copy
import trimesh
import trimesh.transformations as tra
import json
import numpy as np
import pyrender
import time
import torch
import pickle
import pycocotools.mask
import cv2

from manopth.manolayer import ManoLayer
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from tabulate import tabulate

from dex_ycb_toolkit.factory import get_dataset
from dex_ycb_toolkit.logging import get_logger

bop_toolkit_root = os.path.join(os.path.dirname(__file__), "..", "bop_toolkit")
sys.path.append(bop_toolkit_root)

from bop_toolkit_lib import inout

_RADIUS = [0.05]
_ANGLES = [15]
_DIST_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]

_OBJ_MESH_GT_COLOR = (1.0, 1.0, 1.0, 0.6)
_HAND_MESH_GT_COLOR = (0.47, 0.29, 0.21, 0.60)
_HAND_MESH_PC_COLOR = (0.01, 0.02, 0.37, 1.00)

_COVERED_GRASP_COLOR = (0.00, 1.00, 0.04, 1.00)
_COLLIDE_GRASP_COLOR = (0.62, 0.02, 0.01, 1.00)
_FAILURE_GRASP_COLOR = (0.7, 0.7, 0.7, 0.8)


class GraspEvaluator():
  """Grasp evaluator."""
  radius = _RADIUS
  angles = _ANGLES
  dist_thresholds = _DIST_THRESHOLDS

  def __init__(self, name):
    """Constructor.

    Args:
      name: Dataset name. E.g., 's0_test'.
    """
    self._name = name

    self._dataset = get_dataset(self._name)

    self._ycb_meshes = {}

    self._ycb_grasp_file = os.path.join("assets",
                                        "ycb_farthest_100_grasps.json")
    with open(self._ycb_grasp_file, 'r') as f:
      self._ycb_grasps = json.load(f)

    self._mano_layer_r = ManoLayer(flat_hand_mean=False,
                                   ncomps=45,
                                   side='right',
                                   mano_root='manopth/mano/models',
                                   use_pca=True)
    self._mano_layer_l = ManoLayer(flat_hand_mean=False,
                                   ncomps=45,
                                   side='left',
                                   mano_root='manopth/mano/models',
                                   use_pca=True)

    self._gripper_mesh_file = os.path.join("assets", "panda_gripper.obj")
    self._gripper_mesh = trimesh.load(self._gripper_mesh_file)

    self._gripper_mesh_vis_file = os.path.join("assets", "panda_tubes.obj")
    self._gripper_mesh_vis = trimesh.load(self._gripper_mesh_vis_file)

    self._gripper_pc_file = os.path.join("assets", "panda_pc.npy")
    gripper_pc = np.load(self._gripper_pc_file, allow_pickle=True)
    gripper_pc = gripper_pc.item()['points'][:100, :3]
    self._gripper_pc = gripper_pc

    self._h = 480
    self._w = 640
    x = np.linspace(0, self._w - 1, self._w)
    y = np.linspace(0, self._h - 1, self._h)
    self._xmap, self._ymap = np.meshgrid(x, y)

    self._default_coverage = {}
    self._default_precision = {}
    for r in _RADIUS:
      for a in _ANGLES:
        for thr in _DIST_THRESHOLDS:
          self._default_coverage.setdefault(r, {}).setdefault(a, {}).setdefault(
              thr, 0.0)
          self._default_precision.setdefault(r, {}).setdefault(a,
                                                               {}).setdefault(
                                                                   thr, 0.0)

    self._tf_to_opengl = np.eye(4)
    self._tf_to_opengl[1, 1] = -1
    self._tf_to_opengl[2, 2] = -1

    self._covered_grasp_material = pyrender.MetallicRoughnessMaterial(
        alphaMode="BLEND",
        doubleSided=True,
        baseColorFactor=_COVERED_GRASP_COLOR,
        metallicFactor=0.0)
    self._collide_grasp_material = pyrender.MetallicRoughnessMaterial(
        alphaMode="BLEND",
        doubleSided=True,
        baseColorFactor=_COLLIDE_GRASP_COLOR,
        metallicFactor=0.0)
    self._failure_grasp_material = pyrender.MetallicRoughnessMaterial(
        alphaMode="BLEND",
        doubleSided=True,
        baseColorFactor=_FAILURE_GRASP_COLOR,
        metallicFactor=0.0)

    self._r = pyrender.OffscreenRenderer(viewport_width=self._w,
                                         viewport_height=self._h)

    self._out_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    self._anno_file = os.path.join(self._out_dir,
                                   "anno_grasp_{}.pkl".format(self._name))
    if os.path.isfile(self._anno_file):
      print('Found Grasp annotation file.')
    else:
      print('Cannot find Grasp annotation file.')
      self._generate_anno_file()

    self._anno = self._load_anno_file()

  def _generate_anno_file(self):
    """Generates the annotation file."""
    print('Generating Grasp annotation file')
    s = time.time()

    anno = {}

    for i in range(len(self._dataset)):
      if (i + 1) in np.floor(np.linspace(0, len(self._dataset), 11))[1:]:
        print('{:3.0f}%  {:6d}/{:6d}'.format(100 * i / len(self._dataset), i,
                                             len(self._dataset)))

      sample = self._dataset[i]

      # Skip samples not in the eval set.
      if not sample['is_grasp_target']:
        continue

      # Skip objects without pre-generated grasps.
      ycb_class = self._dataset.ycb_classes[sample['ycb_ids'][
          sample['ycb_grasp_ind']]]
      if ycb_class not in self._ycb_grasps:
        continue

      label = np.load(sample['label_file'])

      pose_y = label['pose_y'][sample['ycb_grasp_ind']]
      pose_y = np.vstack((pose_y, np.array([[0, 0, 0, 1]])))
      pose_m = label['pose_m']

      if np.all(pose_m == 0.0):
        verts_m = None
        faces_m = None
      else:
        mano_side = sample['mano_side']
        if mano_side == 'right':
          mano_layer = self._mano_layer_r
        if mano_side == 'left':
          mano_layer = self._mano_layer_l

        pose_m = torch.from_numpy(pose_m)
        mano_pose = pose_m[:, 0:48]
        mano_trans = pose_m[:, 48:51]

        mano_betas = np.array(sample['mano_betas'], dtype=np.float32)
        mano_betas = torch.from_numpy(mano_betas).unsqueeze(0)

        verts_m, _ = mano_layer(mano_pose, mano_betas, mano_trans)
        verts_m = verts_m[0] / 1000
        verts_m = verts_m.numpy()
        faces_m = mano_layer.th_faces

      anno[i] = {
          'pose_y': pose_y,
          'verts_m': verts_m,
          'faces_m': faces_m,
      }

    print('# total samples:        {:6d}'.format(len(self._dataset)))
    print('# valid samples:        {:6d}'.format(len(anno)))
    print('# valid samples w hand: {:6d}'.format(
        sum([
            x['verts_m'] is not None and x['faces_m'] is not None
            for x in anno.values()
        ])))

    with open(self._anno_file, 'wb') as f:
      pickle.dump(anno, f)

    e = time.time()
    print('time: {:7.2f}'.format(e - s))

  def _load_anno_file(self):
    """Loads the annotation file.

    Returns:
      A dictionary holding the loaded annotation.
    """
    with open(self._anno_file, 'rb') as f:
      anno = pickle.load(f)

    return anno

  def _load_ycb_mesh(self, ycb_id):
    """Loads the mesh of a YCB object given a YCB object ID.

    Args:
      ycb_id: A YCB object ID.

    Returns:
      A dictionary holding the meshes for visualizing ground-truth pose and
        predicted pose.
    """
    obj_file = self._dataset.obj_file[ycb_id]
    ycb_mesh_pred = trimesh.load(obj_file)
    ycb_mesh_gt = copy.deepcopy(ycb_mesh_pred)
    ycb_mesh_gt.visual = ycb_mesh_gt.visual.to_color()
    ycb_mesh_gt.visual.face_colors = np.tile(_OBJ_MESH_GT_COLOR,
                                             (len(ycb_mesh_gt.faces), 1))
    ycb_mesh = {
        'gt': ycb_mesh_gt,
        'pred': ycb_mesh_pred,
    }
    return ycb_mesh

  def _get_hand_pc_from_det(self, dets, sample, hand_cat_id=22, radius=0.2):
    """Gets hand point cloud from hand detection and depth image.

    Args:
      dets: A dictionary holding the object and hand detections of an image.
      sample: A dictionary holding an image sample.
      hand_cat_id: Category ID for the hand class.
      radius: Radius threshold for filtering points.

    Returns:
      A float64 numpy array of shape [N, 3] containing the hand point cloud.
    """
    if hand_cat_id not in dets:
      # Use empty point cloud if hand is not detected.
      hand_pc = np.zeros((0, 3))
    else:
      # Use hand detection with highest score.
      max_score = 0
      max_score_ind = 0
      for dt_ind, dt in enumerate(dets[hand_cat_id]):
        if dt['score'] > max_score:
          max_score = dt['score']
          max_score_ind = dt_ind

      segmentation = dets[hand_cat_id][max_score_ind]['segmentation']
      mask = pycocotools.mask.decode(segmentation)

      depth_file = sample['depth_file']
      depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
      depth = depth.astype(np.float32) / 1000

      fx = sample['intrinsics']['fx']
      fy = sample['intrinsics']['fy']
      ppx = sample['intrinsics']['ppx']
      ppy = sample['intrinsics']['ppy']

      pt0 = (self._xmap - ppx) * depth / fx
      pt1 = (self._ymap - ppy) * depth / fy
      pt2 = depth

      mask &= depth > 0

      choose = mask.ravel().nonzero()[0]
      pt0 = pt0.ravel()[choose][:, None]
      pt1 = pt1.ravel()[choose][:, None]
      pt2 = pt2.ravel()[choose][:, None]
      hand_pc = np.hstack((pt0, pt1, pt2))

      if len(hand_pc) > 0:
        hand_center = np.median(hand_pc, axis=0, keepdims=True)
        dist = cdist(hand_pc, hand_center).squeeze(axis=1)
        choose = dist < radius
        hand_pc = hand_pc[choose]

    return hand_pc

  def _compute_grasp_coverage(self, samples, gt_poses, neighborhood_radius,
                              neighborhood_angle):
    """Computes coverage rate of two sets of grasps.

    Args:
      samples: A float64 numpy array of shape [S, 7] containing the grasps to
        cover the other set. Each row contains one grasp represented by
        translation and rotation in quaternion (w, x, y, z).
      gt_poses: A float64 numpy array of shape [G, 7] containing the grasps to
        be covered. Each row contains one grasp represented by translation and
        rotation in quaternion (w, x, y, z).
      neighborhood_radius: Radius Threshold.
      neighborhood_angle: A float64 numpy array of shape [] containing the angle
        threshold.

    Returns:
      num_covered_poses: A int64 numpy array of shape [] containing the number
        of covered grasps.
      covered_sample_id: A int32 numpy array of shape [C] containing the indices
        of covered grasps.
    """
    if len(samples) == 0:
      return 0.0, np.array([], dtype=np.int32)

    # Build kdtree.
    samples_tree = cKDTree(samples[:, :3])

    # Find nearest neighbor.
    gt_neighbors_within_radius = samples_tree.query_ball_point(
        gt_poses[:, :3], r=neighborhood_radius)

    # Check orientation distance for these pairs.
    gt_num_neighbors = np.zeros(len(gt_poses))
    gt_ind_neighbors = []
    for i, (p, n) in enumerate(zip(gt_poses, gt_neighbors_within_radius)):
      conj_p = tra.quaternion_conjugate(p[3:])
      angles = [
          tra.rotation_from_matrix(
              tra.quaternion_matrix(
                  tra.quaternion_multiply(conj_p, samples[b, 3:])))[0]
          for b in n
      ]
      within_neighborhood = np.abs(angles) < neighborhood_angle
      num_neighbors = np.sum(within_neighborhood)
      ind_neighbors = np.array(n)[within_neighborhood]
      gt_num_neighbors[i] = num_neighbors
      gt_ind_neighbors.append(ind_neighbors)

    num_covered_poses = np.sum(gt_num_neighbors > 0)
    covered_sample_id = np.unique(np.concatenate(gt_ind_neighbors)).astype(
        np.int32)

    return num_covered_poses, covered_sample_id

  def _visualize(self, fx, fy, cx, cy, obj_mesh_gt, hand_mesh_gt, obj_mesh_pred,
                 hand_pc, pred_grasps, collision_free, covered_grasp_id,
                 im_real, vis_file):
    """Visualizes predicted grasps and saves to a image file.

    Args:
      fx: Focal length in X direction.
      fy: Focal length in Y direction.
      cx: Principal point offset in X direction.
      cy: Principal point offset in Y direction.
      obj_mesh_gt: Ground-truth object mesh.
      hand_mesh_gt: Ground-truth hand mesh.
      obj_mesh_pred: Predicted object mesh.
      hand_pc: A float64 numpy array of shape [N, 3] containing the hand point
        cloud.
      pred_grasps: A list of float64 numpy arrays of shape [4, 4] containing the
        predicted grasps.
      collision_free: A bool numpy array of shape [G] indicating whether each
        predicted grasp is collision free.
      covered_grasp_id: An int32 numpy array of shape [C] containing the indices
        of collision free grasps that are covered.
      im_real: A uint8 numpy array of shape [H, W, 3] containing the color
        image.
      vis_file: Path to the visualization file.
    """
    scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                           ambient_light=np.array([1.0, 1.0, 1.0]))

    camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy)
    scene.add(camera)

    scene.add(pyrender.Mesh.from_trimesh(obj_mesh_gt, smooth=False))
    if hand_mesh_gt is not None:
      scene.add(pyrender.Mesh.from_trimesh(hand_mesh_gt, smooth=False))
    scene.add(pyrender.Mesh.from_trimesh(obj_mesh_pred))

    hand_mesh_pc = trimesh.creation.uv_sphere(radius=0.003)
    hand_mesh_pc.visual.vertex_colors = _HAND_MESH_PC_COLOR
    tfs = np.tile(np.eye(4), (len(hand_pc), 1, 1))
    tfs[:, :3, 3] = hand_pc
    scene.add(pyrender.Mesh.from_trimesh(hand_mesh_pc, poses=tfs))

    for i, grasp in enumerate(pred_grasps):
      if not collision_free[i]:
        material = self._collide_grasp_material
      elif i in np.where(collision_free)[0][covered_grasp_id]:
        material = self._covered_grasp_material
      else:
        material = self._failure_grasp_material

      scene.add(
          pyrender.Mesh.from_trimesh(
              copy.deepcopy(self._gripper_mesh_vis).apply_transform(
                  grasp).apply_transform(self._tf_to_opengl), material))

    im_render, _ = self._r.render(scene)
    im = 0.2 * im_real.astype(np.float32) + 0.8 * im_render[:, :, ::-1].astype(
        np.float32)
    im = im.astype(np.uint8)

    cv2.imwrite(vis_file, im)

  def evaluate(self, bop_res_file, coco_res_file, out_dir=None,
               visualize=False):
    """Evaluates Grasp metrics given a BOP result file and a COCO result file.

    Args:
      bop_res_file: Path to the BOP result file.
      coco_res_file: Path to the COCO result file.
      out_dir: Path to the output directory.
      visualize: Whether to run visualization.

    Returns:
      A dictionary of results.
    """
    if out_dir is None:
      out_dir = self._out_dir

    bop_res_name = os.path.splitext(os.path.basename(bop_res_file))[0]
    coco_res_name = os.path.splitext(os.path.basename(coco_res_file))[0]
    log_file = os.path.join(
        out_dir, "grasp_eval_{}_{}_{}.log".format(self._name, bop_res_name,
                                                  coco_res_name))
    logger = get_logger(log_file)

    grasp_res_file = os.path.join(
        out_dir, "grasp_res_{}_{}_{}.json".format(self._name, bop_res_name,
                                                  coco_res_name))
    if visualize:
      grasp_vis_dir = os.path.join(
          out_dir, "grasp_vis_{}_{}_{}".format(self._name, bop_res_name,
                                               coco_res_name))
      os.makedirs(grasp_vis_dir, exist_ok=True)

    ests = inout.load_bop_results(bop_res_file)
    ests_org = {}
    for est in ests:
      ests_org.setdefault(est['scene_id'],
                          {}).setdefault(est['im_id'],
                                         {}).setdefault(est['obj_id'],
                                                        []).append(est)

    with open(coco_res_file, 'r') as f:
      dets = json.load(f)
    dets_org = {}
    for det in dets:
      dets_org.setdefault(det['image_id'],
                          {}).setdefault(det['category_id'], []).append(det)

    m = trimesh.collision.CollisionManager()

    logger.info('Running evaluation')

    results = []

    for ind, i in enumerate(self._anno):
      sample = self._dataset[i]

      ycb_id = sample['ycb_ids'][sample['ycb_grasp_ind']]
      ycb_class = self._dataset.ycb_classes[ycb_id]

      # Handle grasped object not being detected.
      scene_id, im_id = self._dataset.get_bop_id_from_idx(i)
      if scene_id not in ests_org or im_id not in ests_org[
          scene_id] or ycb_id not in ests_org[scene_id][im_id]:
        results.append({
            'coverage': self._default_coverage,
            'precision': self._default_precision,
        })
        continue

      # Load YCB mesh.
      if ycb_id not in self._ycb_meshes:
        self._ycb_meshes[ycb_id] = self._load_ycb_mesh(ycb_id)

      # Get ground-truth object mesh.
      obj_pose_gt = self._anno[i]['pose_y']
      obj_mesh_gt = copy.deepcopy(self._ycb_meshes[ycb_id]['gt'])
      obj_mesh_gt.apply_transform(obj_pose_gt)
      m.add_object('gt_obj', obj_mesh_gt)

      # Get ground-truth hand mesh.
      hand_verts = self._anno[i]['verts_m']
      hand_faces = self._anno[i]['faces_m']
      if hand_verts is not None and hand_faces is not None:
        hand_mesh_gt = trimesh.Trimesh(hand_verts, hand_faces)
        if visualize:
          hand_mesh_gt.visual.face_colors = np.tile(
              _HAND_MESH_GT_COLOR, (len(hand_mesh_gt.faces), 1))
      else:
        # Leave hand out of evaluation if ground-truth is missing.
        hand_mesh_gt = None
      if hand_mesh_gt is not None:
        m.add_object('gt_hand', hand_mesh_gt)

      # Calculate ground-truth grasps based on ground-truth object and hand mesh.
      gt_grasps_q = []
      for grasp_o in self._ycb_grasps[ycb_class]:
        grasp_w = np.matmul(obj_pose_gt, grasp_o)
        hit = m.in_collision_single(self._gripper_mesh, transform=grasp_w)
        if not hit:
          q = tra.quaternion_from_matrix(grasp_w, isprecise=True)
          t = grasp_w[:3, 3]
          g = np.hstack((t, q))
          gt_grasps_q.append(g)

      m.remove_object('gt_obj')
      if hand_mesh_gt is not None:
        m.remove_object('gt_hand')

      # Get predicted object mesh.
      ests_sorted = sorted(ests_org[scene_id][im_id][ycb_id],
                           key=lambda e: e['score'],
                           reverse=True)
      est = ests_sorted[0]
      obj_pose_pred = np.eye(4)
      obj_pose_pred[:3, :3] = est['R']
      obj_pose_pred[:3, 3] = est['t'][:, 0] / 1000
      obj_mesh_pred = copy.deepcopy(self._ycb_meshes[ycb_id]['pred'])
      obj_mesh_pred.apply_transform(obj_pose_pred)
      m.add_object('pred_obj', obj_mesh_pred)

      # Get predicted hand point cloud.
      if hand_mesh_gt is not None and i in dets_org:
        hand_pc = self._get_hand_pc_from_det(dets_org[i], sample)
      else:
        hand_pc = np.zeros((0, 3))

      # Calculate predicted grasps based on predicted object mesh and hand point cloud.
      pred_grasps_m = {thr: [] for thr in _DIST_THRESHOLDS}
      pred_grasps_q = {thr: [] for thr in _DIST_THRESHOLDS}
      for grasp_o in self._ycb_grasps[ycb_class]:
        grasp_w = np.matmul(obj_pose_pred, grasp_o)
        hit = m.in_collision_single(self._gripper_mesh, transform=grasp_w)
        if not hit:
          r = grasp_w[:3, :3]
          t = grasp_w[:3, 3]
          gripper_pc = np.matmul(self._gripper_pc, r.T) + t
          if len(hand_pc) == 0:
            min_dist = max(_DIST_THRESHOLDS) + 1
          else:
            min_dist = cdist(gripper_pc, hand_pc).min()

          for thr in _DIST_THRESHOLDS:
            if min_dist > thr:
              pred_grasps_m[thr].append(grasp_w)
              q = tra.quaternion_from_matrix(grasp_w, isprecise=True)
              t = grasp_w[:3, 3]
              g = np.hstack((t, q))
              pred_grasps_q[thr].append(g)

      m.remove_object('pred_obj')

      m.add_object('gt_obj', obj_mesh_gt)
      if hand_mesh_gt is not None:
        m.add_object('gt_hand', hand_mesh_gt)

      if visualize:
        color_file = sample['color_file']
        color = cv2.imread(color_file)

        obj_mesh_gt.apply_transform(self._tf_to_opengl)
        if hand_mesh_gt is not None:
          hand_mesh_gt.apply_transform(self._tf_to_opengl)
        obj_mesh_pred.apply_transform(self._tf_to_opengl)
        hand_pc[:, 1:] *= -1

      # Compute coverage and precision.
      coverage = copy.deepcopy(self._default_coverage)
      precision = copy.deepcopy(self._default_precision)

      if len(gt_grasps_q) > 0:
        for r in _RADIUS:
          for a in _ANGLES:
            for thr in _DIST_THRESHOLDS:
              # Check collision with ground-trtuh object and hand mesh.
              collision_free = np.ones(len(pred_grasps_m[thr]), dtype=bool)
              for g_ind, g in enumerate(pred_grasps_m[thr]):
                hit = m.in_collision_single(self._gripper_mesh, transform=g)
                if hit:
                  collision_free[g_ind] = 0

              if collision_free.sum() > 0:
                num_covered_gt_grasps, covered_pred_grasp_id = self._compute_grasp_coverage(
                    np.array(pred_grasps_q[thr])[collision_free, :],
                    np.array(gt_grasps_q), r, np.deg2rad(a))

                num_covered_pred_grasps, _ = self._compute_grasp_coverage(
                    np.array(gt_grasps_q),
                    np.array(pred_grasps_q[thr])[collision_free, :], r,
                    np.deg2rad(a))

                coverage[r][a][thr] = num_covered_gt_grasps / len(gt_grasps_q)
                precision[r][a][thr] = num_covered_pred_grasps / len(
                    pred_grasps_q[thr])

              if visualize:
                vis_dir = os.path.join(
                    grasp_vis_dir,
                    "radius_{:4.2f}.angle_{:2d}.min-dist-threshold_{:4.2f}".
                    format(r, a, thr))
                os.makedirs(vis_dir, exist_ok=True)
                vis_file = os.path.join(vis_dir, "{:06d}.jpg".format(i))
                self._visualize(sample['intrinsics']['fx'],
                                sample['intrinsics']['fy'],
                                sample['intrinsics']['ppx'],
                                sample['intrinsics']['ppy'], obj_mesh_gt,
                                hand_mesh_gt, obj_mesh_pred, hand_pc,
                                pred_grasps_m[thr], collision_free,
                                covered_pred_grasp_id, color, vis_file)

      results.append({
          'coverage': coverage,
          'precision': precision,
      })

      logger.info('{:04d}/{:04d}  {:6d}  {:21s}  # gt grasps: {:3d}'.format(
          ind + 1, len(self._anno), i, ycb_class, len(gt_grasps_q)))

      m.remove_object('gt_obj')
      if hand_mesh_gt is not None:
        m.remove_object('gt_hand')

    tabular_data = []
    for r in _RADIUS:
      for a in _ANGLES:
        for thr in _DIST_THRESHOLDS:
          coverage = np.mean([x['coverage'][r][a][thr] for x in results])
          precision = np.mean([x['precision'][r][a][thr] for x in results])
          tabular_data.append([r, a, thr, coverage, precision])
    metrics = [
        'radius (m)', 'angle (deg)', 'dist th (m)', 'coverage', 'precision'
    ]
    table = tabulate(tabular_data,
                     headers=metrics,
                     tablefmt='pipe',
                     floatfmt='.4f',
                     numalign='right')
    logger.info('Results: \n' + table)

    with open(grasp_res_file, 'w') as f:
      json.dump(results, f)

    logger.info('Evaluation complete.')

    return results
