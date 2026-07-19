"""Why does the offline flip-discrim (0.58) not match the training log (0.955)?

Calls the model's OWN forward (not a re-implementation) on a pretrain-shaped
EgoBrain batch and reads info['diag_flip_discrim_acc'] directly, under the four
conditions that differ between my offline probe and the training loop:
  masked (mask_ratio 0.5) x {train(), eval()}
  unmasked              x {train(), eval()}
Whichever reproduces ~0.955 tells us what the diagnostic actually depends on.
"""
import argparse
import functools
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

from utils.util import load_pretrain_checkpoint, apply_arch_params, generate_mask  # noqa
from models import get_model  # noqa
from datasets.egobrain_dataset import EgoBrainDataset, collate_egobrain  # noqa

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='outputs/wm-new-gradneg/epoch10_loss2.3018484115600586.pth')
ap.add_argument('--n_batches', type=int, default=12)
ap.add_argument('--batch_size', type=int, default=64)
a = ap.parse_args()

sd, saved = load_pretrain_checkpoint(a.ckpt)
p = argparse.Namespace(model='Align', dropout=0.1)
apply_arch_params(p, saved)
enc = get_model(p, None, None)
miss, _ = enc.load_state_dict(
    {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')},
    strict=False)
assert not miss, miss[:5]
enc = enc.to(DEV)
print(f"mask_ratio(saved)={saved.get('mask_ratio')} n_windows={saved.get('egobrain_n_windows')}",
      flush=True)

ds = EgoBrainDataset(
    data_dir=os.path.join(REPO, 'data/EgoBrain'),
    subjects=[f'P{i:04d}' for i in range(1, 41)],
    in_dim=p.in_dim, n_windows=saved.get('egobrain_n_windows', 6),
    window_s=1.0, stride_s=0.2, clip_s=4.0, erp_latency_s=-0.15,
    fs_out=200, max_channels=32, load_frames=True, frame_size=224,
    use_frame_grid=True, frame_grid_s=0.2, use_grid_embeddings=True,
    temporal_jitter=False, motion_resample=False)
g = torch.Generator().manual_seed(0)
dl = torch.utils.data.DataLoader(
    ds, batch_size=a.batch_size, shuffle=True, generator=g, num_workers=0,
    collate_fn=functools.partial(collate_egobrain, frame_objective=True))

acc = {k: [] for k in ('mask_train', 'mask_eval', 'nomask_train', 'nomask_eval')}
it = iter(dl)
for i in range(a.n_batches):
    b = next(it)
    b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
    b['timeseries'] = b['timeseries'] / 100.0
    x = b['timeseries']
    B, C, N, _ = x.shape
    # generate_mask(bz, ch_num, patch_num, ...) -- pretrain_trainer.py:274-278
    m = generate_mask(B, C, N, mask_ratio=float(saved.get('mask_ratio', 0.5)),
                      device=DEV)
    for cond in acc:
        use_mask = cond.startswith('mask')
        enc.train(cond.endswith('train'))
        with torch.no_grad():
            _, info = enc(dict(b), mask=(m if use_mask else None))
        v = info.get('diag_flip_discrim_acc')
        if v is not None:
            acc[cond].append(float(v))

print()
print("model's OWN diag_flip_discrim_acc (alignment.py:2255), by condition:")
for k, v in acc.items():
    if v:
        t = torch.tensor(v)
        print(f"  {k:14s}  {t.mean():.4f}  (sd {t.std():.4f}, n={len(v)} batches)")
print()
print("training log (wandb tcnfsvt7, ep10): 0.9554")
