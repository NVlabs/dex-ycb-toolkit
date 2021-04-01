# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""BOP evaluator."""

import os
import sys
import numpy as np
import subprocess
import itertools

from collections import defaultdict
from tabulate import tabulate

from dex_ycb_toolkit.factory import get_dataset
from dex_ycb_toolkit.logging import get_logger

bop_toolkit_root = os.path.join(os.path.dirname(__file__), "..", "bop_toolkit")
sys.path.append(bop_toolkit_root)

from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import score

# See dataset_info.md in http://ptak.felk.cvut.cz/6DB/public/bop_datasets/ycbv_base.zip.
_BOP_TRANSLATIONS = {
    '002_master_chef_can':   [  1.3360,  -0.5000,   3.5105],
    '003_cracker_box':       [  0.5575,   1.7005,   4.8050],
    '004_sugar_box':         [ -0.9520,   1.4670,   4.3645],
    '005_tomato_soup_can':   [ -0.0240,  -1.5270,   8.4035],
    '006_mustard_bottle':    [  1.2995,   2.4870, -11.8290],
    '007_tuna_fish_can':     [ -0.1565,   0.1150,   4.2625],
    '008_pudding_box':       [  1.1645,  -4.2015,   3.1190],
    '009_gelatin_box':       [  1.4460,  -0.5915,   3.6085],
    '010_potted_meat_can':   [  2.4195,   0.3075,   8.0715],
    '011_banana':            [-18.6730,  12.1915,  -1.4635],
    '019_pitcher_base':      [  5.3370,   5.8855,  25.6115],
    '021_bleach_cleanser':   [  4.9290,  -2.4800, -13.2920],
    '024_bowl':              [ -0.2270,   0.7950,  -2.9675],
    '025_mug':               [ -8.4675,  -0.6995,  -1.6145],
    '035_power_drill':       [  9.0710,  20.9360,  -2.1190],
    '036_wood_block':        [  1.4265,  -2.5305,  17.1890],
    '037_scissors':          [  7.0535, -28.1320,   0.0420],
    '040_large_marker':      [  0.0460,  -2.1040,   0.3500],
    '051_large_clamp':       [ 10.5180,  -1.9640,  -0.4745],
    '052_extra_large_clamp': [ -0.3950, -10.4130,   0.1620],
    '061_foam_brick':        [ -0.0805,   0.0805,  -8.2435],
}


class BOPEvaluator():
  """BOP evaluator."""

  def __init__(self, name):
    """Constructor.

    Args:
      name: Dataset name. E.g., 's0_test'.
    """
    self._name = name

    self._dataset = get_dataset(self._name)

    self._setup = self._name.split('_')[0]
    self._split = self._name.split('_')[1]

    self._out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    self._bop_dir = os.path.join(self._dataset.data_dir, "bop")

    self._p = {
        'errors': [
            {
                'n_top': -1,
                'type': 'vsd',
                'vsd_delta': 15,
                'vsd_taus': list(np.arange(0.05, 0.51, 0.05)),
                'correct_th': [[th] for th in np.arange(0.05, 0.51, 0.05)]
            },
            {
                'n_top': -1,
                'type': 'mssd',
                'correct_th': [[th] for th in np.arange(0.05, 0.51, 0.05)]
            },
            {
                'n_top': -1,
                'type': 'mspd',
                'correct_th': [[th] for th in np.arange(5, 51, 5)]
            },
        ],
        'visib_gt_min': -1,
    }

    dp_split = dataset_params.get_split_params(self._bop_dir, self._setup,
                                               self._split)
    dp_model = dataset_params.get_model_params(self._bop_dir,
                                               self._setup,
                                               model_type='eval')
    self._scene_ids = dp_split['scene_ids']
    self._obj_ids = dp_model['obj_ids']

    self._grasp_id = defaultdict(lambda: {})
    for i in range(len(self._dataset)):
      sample = self._dataset[i]
      scene_id, im_id = self._dataset.get_bop_id_from_idx(i)
      obj_id = sample['ycb_ids'][sample['ycb_grasp_ind']]
      self._grasp_id[scene_id][im_id] = obj_id

  def _convert_pose_to_bop(self, est):
    """Converts pose from DexYCB models to BOP YCBV models.

    Args:
      est: A dictionary holding a single pose estimate.

    Returns:
      A dictionary holding the converted pose.
    """
    est['t'] -= np.dot(
        est['R'],
        _BOP_TRANSLATIONS[self._dataset.ycb_classes[est['obj_id']]]).reshape(
            3, 1)
    return est

  def _derive_bop_results(self, out_dir, result_name, grasp_only, logger):
    """Derives BOP results.

    Args:
      out_dir: Path to the output directory.
      result_name: BOP result name. Should be the name of a folder under out_dir
        that contains output from BOP evaluation.
      grasp_only: Whether to derive results on grasped objects only.
      logger: Logger.

    Returns:
      A dictionary holding the results.
    """
    if grasp_only:
      set_str = 'grasp only'
    else:
      set_str = 'all'

    logger.info('Deriving results for *{}*'.format(set_str))

    average_recalls = {}
    average_recalls_obj = defaultdict(lambda: {})

    for error in self._p['errors']:

      error_dir_paths = {}
      if error['type'] == 'vsd':
        for vsd_tau in error['vsd_taus']:
          error_sign = misc.get_error_signature(error['type'],
                                                error['n_top'],
                                                vsd_delta=error['vsd_delta'],
                                                vsd_tau=vsd_tau)
          error_dir_paths[error_sign] = os.path.join(result_name, error_sign)
      else:
        error_sign = misc.get_error_signature(error['type'], error['n_top'])
        error_dir_paths[error_sign] = os.path.join(result_name, error_sign)

      recalls = []
      recalls_obj = defaultdict(lambda: [])

      for error_sign, error_dir_path in error_dir_paths.items():
        for correct_th in error['correct_th']:

          score_sign = misc.get_score_signature(correct_th,
                                                self._p['visib_gt_min'])
          matches_filename = "matches_{}.json".format(score_sign)
          matches_path = os.path.join(out_dir, error_dir_path, matches_filename)

          matches = inout.load_json(matches_path)

          if grasp_only:
            matches = [
                m for m in matches
                if m['obj_id'] == self._grasp_id[m['scene_id']][m['im_id']]
            ]

          scores = score.calc_localization_scores(self._scene_ids,
                                                  self._obj_ids,
                                                  matches,
                                                  error['n_top'],
                                                  do_print=False)

          recalls.append(scores['recall'])
          for i, r in scores['obj_recalls'].items():
            recalls_obj[i].append(r)

      average_recalls[error['type']] = np.mean(recalls)
      for i, r in recalls_obj.items():
        average_recalls_obj[i][error['type']] = np.mean(r)

    results = {i: r * 100 for i, r in average_recalls.items()}
    results['mean'] = np.mean(
        [results['vsd'], results['mssd'], results['mspd']])

    keys, values = tuple(zip(*results.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt='pipe',
        floatfmt='.3f',
        stralign='center',
        numalign='center',
    )
    logger.info('Evaluation results for *{}*: \n'.format(set_str) + table)

    results_per_object = {}
    for i, v in average_recalls_obj.items():
      res = {k: r * 100 for k, r in v.items()}
      res['mean'] = np.mean([res['vsd'], res['mssd'], res['mspd']])
      results_per_object[self._dataset.ycb_classes[i]] = res

    n_cols = 5
    results_tuple = [(k, v['vsd'], v['mssd'], v['mspd'], v['mean'])
                     for k, v in results_per_object.items()]
    results_flatten = list(itertools.chain(*results_tuple))
    results_2d = itertools.zip_longest(
        *[results_flatten[i::n_cols] for i in range(n_cols)])
    table = tabulate(
        results_2d,
        tablefmt='pipe',
        floatfmt='.3f',
        headers=['object', 'vsd', 'mssd', 'mspd', 'mean'] * (n_cols // 5),
        numalign='right',
    )
    logger.info('Per-object scores for *{}*: \n'.format(set_str) + table)

    results['per_obj'] = results_per_object

    return results

  def evaluate(self, res_file, out_dir=None, renderer_type='python'):
    """Evaluates BOP metrics given a result file.

    Args:
      res_file: Path to the result file.
      out_dir: Path to the output directory.
      renderer_type: Renderer type. 'python' or 'cpp'.

    Returns:
      A dictionary holding the results.

    Raises:
      RuntimeError: If BOP evaluation failed.
    """
    if out_dir is None:
      out_dir = self._out_dir

    ests = inout.load_bop_results(res_file)
    ests = [self._convert_pose_to_bop(est) for est in ests]
    res_name = os.path.splitext(os.path.basename(res_file))[0]
    bop_res_name = 'bop-{}_{}-{}'.format(res_name.replace('_', '-'),
                                         self._setup, self._split)
    bop_res_file = os.path.join(out_dir, "{}.csv".format(bop_res_name))
    inout.save_bop_results(bop_res_file, ests)

    eval_cmd = [
        'python',
        os.path.join('scripts', 'eval_bop19.py'),
        '--renderer_type={}'.format(renderer_type),
        '--result_filenames={}'.format(bop_res_file),
        '--results_path={}'.format(out_dir),
        '--eval_path={}'.format(out_dir),
    ]
    cwd = "bop_toolkit"
    env = os.environ.copy()
    env['PYTHONPATH'] = "."
    env['BOP_PATH'] = self._bop_dir

    if subprocess.run(eval_cmd, cwd=cwd, env=env).returncode != 0:
      raise RuntimeError('BOP evaluation failed.')

    log_file = os.path.join(out_dir,
                            "bop_eval_{}_{}.log".format(self._name, res_name))
    logger = get_logger(log_file)

    results = {}
    results['all'] = self._derive_bop_results(out_dir, bop_res_name, False,
                                              logger)
    results['grasp_only'] = self._derive_bop_results(out_dir, bop_res_name,
                                                     True, logger)

    logger.info('Evaluation complete.')

    return results
