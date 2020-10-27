import json

from lib.factory import get_dataset

for setup in ('s0', 's1', 's2', 's3'):
  for split in ('train', 'val', 'test'):
    name = '{}_{}'.format(setup, split)
    print('dataset name: {}'.format(name))

    dataset = get_dataset(name)

    print('dataset size: {}'.format(len(dataset)))

    sample = dataset[999]
    print('1000th sample:')
    print(json.dumps(sample, indent=4))
