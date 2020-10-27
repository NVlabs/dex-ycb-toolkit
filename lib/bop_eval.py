import os
import sys
import numpy as np
import subprocess

from lib.factory import get_dataset

bop_toolkit_root = os.path.join(os.path.dirname(__file__), "..", "bop_toolkit")
sys.path.append(bop_toolkit_root)

from bop_toolkit_lib import inout

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

  def __init__(self, name):
    self._name = name

    self._dataset = get_dataset(self._name)

    self._eval_dir = os.path.join(os.path.dirname(__file__), "..", "eval")
    self._bop_dir = os.path.join(self._dataset.data_dir, "bop")

  def _convert_pose_to_bop(self, est):
    est['t'] *= 1000
    est['t'] -= np.dot(
        est['R'],
        _BOP_TRANSLATIONS[self._dataset.ycb_classes[est['obj_id']]]).reshape(
            3, 1)
    return est

  def evaluate(self, res_file, renderer_type='python'):
    ests = inout.load_bop_results(res_file)
    ests = [self._convert_pose_to_bop(est) for est in ests]
    bop_res_file = os.path.join(
        self._eval_dir, "bop-res_{}-{}.csv".format(*self._name.split('_')))
    inout.save_bop_results(bop_res_file, ests)

    eval_cmd = [
        'python',
        os.path.join('scripts', 'eval_bop19.py'),
        '--renderer_type={}'.format(renderer_type),
        '--result_filenames={}'.format(bop_res_file),
        '--results_path={}'.format(self._eval_dir),
        '--eval_path={}'.format(self._eval_dir),
    ]
    cwd = "bop_toolkit"
    env = os.environ.copy()
    env['PYTHONPATH'] = "."
    env['BOP_PATH'] = self._bop_dir

    if subprocess.run(eval_cmd, cwd=cwd, env=env).returncode != 0:
      raise RuntimeError('BOP evaluation failed.')
