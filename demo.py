from lib.factory import get_dataset

for setup in ('s0', 's1', 's2', 's3'):
  for split in ('train', 'val'):
    name = '{}_{}'.format(setup, split)
    print('dataset name: {}'.format(name))

    dataset = get_dataset(name)
    
    print('dataset size: {}'.format(len(dataset)))

    x = dataset[0]
    print('1st sample:')
    print(x[0])
    print(x[1])
    print(x[2])

    x = dataset[999]
    print('1000th sample:')
    print(x[0])
    print(x[1])
    print(x[2])
