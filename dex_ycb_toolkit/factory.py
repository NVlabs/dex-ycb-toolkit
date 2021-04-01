# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Factory method for easily getting datasets by name."""

from .dex_ycb import DexYCBDataset

_sets = {}

for setup in ('s0', 's1', 's2', 's3'):
  for split in ('train', 'val', 'test'):
    name = '{}_{}'.format(setup, split)
    _sets[name] = (lambda setup=setup, split=split: DexYCBDataset(setup, split))


def get_dataset(name):
  """Gets a dataset by name.

  Args:
    name: Dataset name. E.g., 's0_test'.

  Returns:
    A dataset.

  Raises:
    KeyError: If name is not supported.
  """
  if name not in _sets:
    raise KeyError('Unknown dataset name: {}'.format(name))
  return _sets[name]()
