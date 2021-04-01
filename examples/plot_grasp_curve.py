# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of plotting grasp precision-coverage curve."""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate

from dex_ycb_toolkit.grasp_eval import GraspEvaluator

res_dir = os.path.join(os.path.dirname(__file__), "..", "results")

methods = [
    {
        'name': 'PoseCNN RGB',
        'path': os.path.join(res_dir, "cvpr2021_results", "grasp_res_{name}_bop_posecnn_{name}_coco_maskrcnn_{name}.json"),
        'linestyle': '-',
        'marker': '.',
    },
    {
        'name': 'PoseCNN + Depth',
        'path': os.path.join(res_dir, "cvpr2021_results", "grasp_res_{name}_bop_posecnn_{name}_refined_coco_maskrcnn_{name}.json"),
        'linestyle': '-',
        'marker': 'o',
    },
    {
        'name': 'DeepIM RGB',
        'path': os.path.join(res_dir, "cvpr2021_results", "grasp_res_{name}_bop_deepim_{name}_COLOR_coco_maskrcnn_{name}.json"),
        'linestyle': '-',
        'marker': 'v',
    },
    {
        'name': 'DeepIM RGB-D',
        'path': os.path.join(res_dir, "cvpr2021_results", "grasp_res_{name}_bop_deepim_{name}_RGBD_coco_maskrcnn_{name}.json"),
        'linestyle': '-',
        'marker': '^',
    },
    {
        'name': 'PoseRBPF RGB',
        'path': os.path.join(res_dir, "cvpr2021_results", "grasp_res_{name}_bop_poserbpf_{name}_rgb_coco_maskrcnn_{name}.json"),
        'linestyle': '-',
        'marker': '>',
    },
    {
        'name': 'PoseRBPF RGB-D',
        'path': os.path.join(res_dir, "cvpr2021_results", "grasp_res_{name}_bop_poserbpf_{name}_rgbd_coco_maskrcnn_{name}.json"),
        'linestyle': '-',
        'marker': '*',
    },
    {
        'name': 'CosyPose RGB',
        'path': os.path.join(res_dir, "cvpr2021_results", "grasp_res_{name}_bop_cosypose_{name}_coco_maskrcnn_{name}.json"),
        'linestyle': '-',
        'marker': 'D',
    },
]


def parse_args():
  parser = argparse.ArgumentParser(description='Plot grasp precision-coverage curve.')
  parser.add_argument('--name', help='Dataset name', default='s1_test', type=str)
  args = parser.parse_args()
  return args


def load_grasp_res_file(grasp_res_file):
  """Loads a Grasp result file.

  Args:
    grasp_res_file: Path to the Grasp result file.

  Returns:
    A dictionary holding the loaded Grasp results.
  """
  def _convert_keys_to_float(x):
    def _try_convert(k):
      try:
        return float(k)
      except ValueError:
        return k
    return {_try_convert(k): v for k, v in x.items()}

  with open(grasp_res_file, 'r') as f:
    results = json.load(f, object_hook=lambda x: _convert_keys_to_float(x))

  return results


def main():
  args = parse_args()
  print('Dataset name: {}'.format(args.name))

  plt.figure()

  for m in methods:
    print('Method: {}'.format(m['name']).format(name=args.name))

    results = load_grasp_res_file(m['path'].format(name=args.name))

    coverages = []
    precisions = []
    tabular_data = []
    for r in GraspEvaluator.radius:
      for a in GraspEvaluator.angles:
        for thr in GraspEvaluator.dist_thresholds:
          c = np.mean([x['coverage'][r][a][thr] for x in results])
          p = np.mean([x['precision'][r][a][thr] for x in results])
          coverages.append(c)
          precisions.append(p)
          tabular_data.append([r, a, thr, c, p])
    metrics = [
        'radius (m)', 'angle (deg)', 'dist th (m)', 'coverage', 'precision'
    ]
    table = tabulate(tabular_data,
                     headers=metrics,
                     tablefmt='pipe',
                     floatfmt='.4f',
                     numalign='right')
    print('Results: \n' + table)
    
    plt.plot(coverages,
             precisions,
             label=m['name'],
             linestyle=m['linestyle'],
             marker=m['marker'])

  plt.xlabel('Coverage')
  plt.ylabel('Precision')
  plt.xlim(0, plt.xlim()[1])
  plt.ylim(0, plt.ylim()[1])
  plt.grid()
  plt.legend()
  plt.tight_layout()

  grasp_curve_file = os.path.join(
      res_dir, "grasp_precision_coverage_{name}.pdf".format(name=args.name))
  print('Saving figure to {}'.format(grasp_curve_file))
  plt.savefig(grasp_curve_file)

  plt.show()


if __name__ == '__main__':
  main()
