"""CPU smoke check: pull one real batch through a downstream loader and assert
the collated tensor shapes/dtypes match what the model expects. No GPU, no
checkpoint, no model build."""
import argparse
import importlib
import os
import sys
import types

# Run from scripts/ puts the local `datasets/` package behind the installed
# HuggingFace `datasets`; force the project root onto the front of sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument('--loader', required=True, help='datasets module, e.g. weibo2014_dataset')
ap.add_argument('--datasets_dir', required=True)
ap.add_argument('--channels', type=int, required=True)
ap.add_argument('--classes', type=int, required=True)
ap.add_argument('--windows', type=int, default=4)
a = ap.parse_args()

mod = importlib.import_module(f'datasets.{a.loader}')
params = types.SimpleNamespace(datasets_dir=a.datasets_dir, batch_size=2,
                               use_SmallerToken=False)
dl = mod.LoadDataset(params).get_data_loader()
for split in ('train', 'val', 'test'):
    b = next(iter(dl[split]))
    x, y = b['x'], b['y']
    assert tuple(x.shape) == (2, a.channels, a.windows, 200), (split, x.shape)
    assert x.dtype.is_floating_point, x.dtype
    assert int(y.min()) >= 0 and int(y.max()) < a.classes, (split, y.tolist())
    assert b['ch_coords'].shape == (2, a.channels, 3), b['ch_coords'].shape
    print(f'[{split}] x={tuple(x.shape)} {x.dtype}  y={y.tolist()}  coords={tuple(b["ch_coords"].shape)}')
print('DATALOAD_SMOKE_OK')
