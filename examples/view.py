import argparse
import pyglet

import _init_paths
import dataloader
import window


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
  parser.add_argument('--use-cache', action='store_true', default=False)
  parser.add_argument('--load-ycb', action='store_true', default=False)
  parser.add_argument('--src-ycb',
                      help='Source of the YCB pose',
                      default='full',
                      type=str,
                      choices=['pcnn', 'fuse', 'full', 'release'])
  parser.add_argument('--load-mano', action='store_true', default=False)
  parser.add_argument('--src-mano',
                      help='Source of the MANO pose',
                      default='full',
                      type=str,
                      choices=['kpts', 'full', 'release'])
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()

  loader = dataloader.DataLoader(args.name,
                                 device=args.device,
                                 preload=(not args.no_preload),
                                 use_cache=args.use_cache,
                                 load_ycb=args.load_ycb,
                                 src_ycb=args.src_ycb,
                                 load_mano=args.load_mano,
                                 src_mano=args.src_mano)
  w = window.Window(loader)

  def run(dt):
    w.update()

  pyglet.clock.schedule(run)
  pyglet.app.run()
