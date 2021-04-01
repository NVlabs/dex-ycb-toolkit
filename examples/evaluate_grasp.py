# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of running Grasp evaluation."""

import argparse
import os

from dex_ycb_toolkit.grasp_eval import GraspEvaluator


def parse_args():
  parser = argparse.ArgumentParser(description='Run grasp evaluation.')
  parser.add_argument('--name', help='Dataset name', default=None, type=str)
  parser.add_argument('--bop_res_file',
                      help='Path to BOP result file',
                      default=None,
                      type=str)
  parser.add_argument('--coco_res_file',
                      help='Path to COCO result file',
                      default=None,
                      type=str)
  parser.add_argument('--out_dir',
                      help='Directory to save eval output',
                      default=None,
                      type=str)
  parser.add_argument('--visualize', action='store_true', default=False)
  args = parser.parse_args()
  return args


def main():
  args = parse_args()

  if args.name is None and args.coco_res_file is None and args.bop_res_file is None:
    args.name = 's0_test'
    args.bop_res_file = os.path.join(
        os.path.dirname(__file__), "..", "results",
        "example_results_bop_{}.csv".format(args.name))
    args.coco_res_file = os.path.join(
        os.path.dirname(__file__), "..", "results",
        "example_results_coco_{}.json".format(args.name))

  grasp_eval = GraspEvaluator(args.name)
  grasp_eval.evaluate(args.bop_res_file,
                      args.coco_res_file,
                      out_dir=args.out_dir,
                      visualize=args.visualize)


if __name__ == '__main__':
  main()
