# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of viewing a sequence."""

import argparse
import pyglet

from dex_ycb_toolkit.sequence_loader import SequenceLoader
from dex_ycb_toolkit.window import Window


def parse_args():
  parser = argparse.ArgumentParser(
      description='View point cloud and ground-truth hand & object poses in 3D.'
  )
  parser.add_argument('--name',
                      help='Name of the sequence',
                      default=None,
                      type=str)
  parser.add_argument('--device',
                      help='Device for data loader computation',
                      default='cuda:0',
                      type=str)
  parser.add_argument('--no-preload', action='store_true', default=False)
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()

  loader = SequenceLoader(args.name,
                          device=args.device,
                          preload=(not args.no_preload),
                          app='viewer')
  w = Window(loader)

  def run(dt):
    w.update()

  pyglet.clock.schedule(run)
  pyglet.app.run()
