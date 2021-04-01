# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""HPE evaluator."""

import os
import sys
import time
import numpy as np
import pickle

from tabulate import tabulate

from dex_ycb_toolkit.factory import get_dataset
from dex_ycb_toolkit.logging import get_logger

freihand_root = os.path.join(os.path.dirname(__file__), "..", "freihand")
sys.path.append(freihand_root)

from utils.eval_util import EvalUtil
from eval import align_w_scale, curve, createHTML

_AUC_VAL_MIN = 0.0
_AUC_VAL_MAX = 50.0
_AUC_STEPS = 100


class HPEEvaluator():
  """HPE evaluator."""

  def __init__(self, name):
    """Constructor.

    Args:
      name: Dataset name. E.g., 's0_test'.
    """
    self._name = name

    self._dataset = get_dataset(self._name)

    self._out_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    self._anno_file = os.path.join(self._out_dir,
                                   "anno_hpe_{}.pkl".format(self._name))

    if os.path.isfile(self._anno_file):
      print('Found HPE annotation file.')
    else:
      print('Cannot find HPE annotation file.')
      self._generate_anno_file()

    self._anno = self._load_anno_file()

  def _generate_anno_file(self):
    """Generates the annotation file."""
    print('Generating HPE annotation file')
    s = time.time()

    joint_3d_gt = {}

    for i in range(len(self._dataset)):
      if (i + 1) in np.floor(np.linspace(0, len(self._dataset), 11))[1:]:
        print('{:3.0f}%  {:6d}/{:6d}'.format(100 * i / len(self._dataset), i,
                                             len(self._dataset)))

      sample = self._dataset[i]

      label = np.load(sample['label_file'])
      joint_3d = label['joint_3d'].reshape(21, 3)

      if np.all(joint_3d == -1):
        continue

      joint_3d *= 1000

      joint_3d_gt[i] = joint_3d

    print('# total samples: {:6d}'.format(len(self._dataset)))
    print('# valid samples: {:6d}'.format(len(joint_3d_gt)))

    anno = {
        'joint_3d': joint_3d_gt,
    }
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

    anno['joint_3d'] = {
        k: v.astype(np.float64) for k, v in anno['joint_3d'].items()
    }

    return anno

  def _load_results(self, res_file):
    """Loads results from a result file.

    Args:
      res_file: Path to the result file.

    Returns:
      A dictionary holding the loaded results.

    Raises:
      ValueError: If a line in the result file does not have 64 comma-seperated
        elements.
    """
    results = {}
    with open(res_file, 'r') as f:
      for line in f:
        elems = line.split(',')
        if len(elems) != 64:
          raise ValueError(
              'a line does not have 64 comma-seperated elements: {}'.format(
                  line))
        image_id = int(elems[0])
        joint_3d = np.array(elems[1:], dtype=np.float64).reshape(21, 3)
        results[image_id] = joint_3d
    return results

  def evaluate(self, res_file, out_dir=None):
    """Evaluates HPE metrics given a result file.

    Args:
      res_file: Path to the result file.
      out_dir: Path to the output directory.

    Returns:
      A dictionary holding the results.
    """
    if out_dir is None:
      out_dir = self._out_dir

    res_name = os.path.splitext(os.path.basename(res_file))[0]
    log_file = os.path.join(out_dir,
                            "hpe_eval_{}_{}.log".format(self._name, res_name))
    logger = get_logger(log_file)

    res = self._load_results(res_file)

    logger.info('Running evaluation')

    joint_3d_gt = self._anno['joint_3d']

    eval_util_ab = EvalUtil()
    eval_util_rr = EvalUtil()
    eval_util_pa = EvalUtil()

    for i, kpt_gt in joint_3d_gt.items():
      assert i in res, "missing image id in result file: {}".format(i)
      vis = np.ones_like(kpt_gt[:, 0])
      kpt_pred = res[i]

      eval_util_ab.feed(kpt_gt, vis, kpt_pred)
      eval_util_rr.feed(kpt_gt - kpt_gt[0], vis, kpt_pred - kpt_pred[0])
      eval_util_pa.feed(kpt_gt, vis, align_w_scale(kpt_gt, kpt_pred))

    mean_ab, _, auc_ab, pck_ab, thresh_ab = eval_util_ab.get_measures(
        _AUC_VAL_MIN, _AUC_VAL_MAX, _AUC_STEPS)
    mean_rr, _, auc_rr, pck_rr, thresh_rr = eval_util_rr.get_measures(
        _AUC_VAL_MIN, _AUC_VAL_MAX, _AUC_STEPS)
    mean_pa, _, auc_pa, pck_pa, thresh_pa = eval_util_pa.get_measures(
        _AUC_VAL_MIN, _AUC_VAL_MAX, _AUC_STEPS)

    tabular_data = [['absolute', mean_ab, auc_ab],
                    ['root-relative', mean_rr, auc_rr],
                    ['procrustes', mean_pa, auc_pa]]
    metrics = ['alignment', 'MPJPE (mm)', 'AUC']
    table = tabulate(tabular_data,
                     headers=metrics,
                     tablefmt='pipe',
                     floatfmt='.4f',
                     numalign='right')
    logger.info('Results: \n' + table)

    hpe_curve_dir = os.path.join(out_dir,
                                 "hpe_curve_{}_{}".format(self._name, res_name))
    os.makedirs(hpe_curve_dir, exist_ok=True)

    createHTML(hpe_curve_dir, [
        curve(thresh_ab, pck_ab, 'Distance in mm',
              'Percentage of correct keypoints',
              'PCK curve for absolute keypoint error'),
        curve(thresh_rr, pck_rr, 'Distance in mm',
              'Percentage of correct keypoints',
              'PCK curve for root-relative keypoint error'),
        curve(thresh_pa, pck_pa, 'Distance in mm',
              'Percentage of correct keypoints',
              'PCK curve for Procrustes aligned keypoint error'),
    ])

    results = {
        'absolute': {
            'mpjpe': mean_ab,
            'auc': auc_ab
        },
        'root-relative': {
            'mpjpe': mean_rr,
            'auc': auc_rr
        },
        'procrustes': {
            'mpjpe': mean_pa,
            'auc': auc_pa
        },
    }

    logger.info('Evaluation complete.')

    return results
