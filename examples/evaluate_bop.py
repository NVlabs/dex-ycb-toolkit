# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of running BOP evaluation."""

import argparse
import os

from dex_ycb_toolkit.bop_eval import BOPEvaluator


def parse_args():
  parser = argparse.ArgumentParser(description='Run BOP evaluation.')
  parser.add_argument('--name', help='Dataset name', default=None, type=str)
  parser.add_argument('--res_file',
                      help='Path to result file',
                      default=None,
                      type=str)
  parser.add_argument('--out_dir',
                      help='Directory to save eval output',
                      default=None,
                      type=str)
  args = parser.parse_args()
  return args


def main():
  args = parse_args()

  if args.name is None and args.res_file is None:
    args.name = 's0_test'
    args.res_file = os.path.join(os.path.dirname(__file__), "..", "results",
                                 "example_results_bop_{}.csv".format(args.name))

  bop_eval = BOPEvaluator(args.name)
  bop_eval.evaluate(args.res_file, out_dir=args.out_dir, renderer_type='python')


if __name__ == '__main__':
  main()
