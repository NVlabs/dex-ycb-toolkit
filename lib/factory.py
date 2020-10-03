from .dex_ycb import DexYCBDataset

_sets = {}

for setup in ('s0', 's1', 's2', 's3'):
  for split in ('train', 'val', 'test'):
    if setup == 's1' and split == 'test':
      continue
    name = '{}_{}'.format(setup, split)
    _sets[name] = (lambda setup=setup, split=split: DexYCBDataset(setup, split))


def get_dataset(name):
  if name not in _sets:
    raise KeyError('Unknown dataset name: {}'.format(name))
  return _sets[name]()
