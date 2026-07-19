"""GPU-free end-to-end smoke test for a downstream finetune wiring.

Runs the REAL finetune_main setup (argparse -> apply_arch_params from the
checkpoint -> build model_for_<ds> with pretrained weights -> real data loader)
but monkeypatches the Trainer with a CPU stub that pulls one batch, runs a
forward pass, computes CrossEntropyLoss, and does one backward. This exercises
model build + checkpoint load + patch/crop reshape + forward + loss + backward
without needing a GPU / the SLURM queue.

  usage: python scripts/_smoke_cpu_forward.py --downstream Kaya5F \
             --datasets_dir data/preprocessed/kaya5f --num_of_classes 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['WANDB_MODE'] = 'disabled'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

ap = argparse.ArgumentParser()
ap.add_argument('--downstream', required=True)
ap.add_argument('--datasets_dir', required=True)
ap.add_argument('--num_of_classes', type=int, required=True)
ap.add_argument('--foundation_dir',
                default='outputs/wm-dino-dense/epoch10_loss2.1733548641204834.pth')
a = ap.parse_args()

import torch
import torch.nn as nn

sys.argv = [
    'finetune_main.py',
    '--model', 'Align',
    '--downstream_dataset', a.downstream,
    '--datasets_dir', a.datasets_dir,
    '--num_of_classes', str(a.num_of_classes),
    '--foundation_dir', a.foundation_dir,
    '--use_pretrained_weights',
    '--use_initial_segment_only',
    '--batch_size', '2',
    '--num_workers', '0',
    '--model_dir', 'outputs/CSBrain/_smoke_cpu',
]

import finetune_main

finetune_main.wandb.init = lambda *args, **kw: None


class _CPUSmokeTrainer:
    def __init__(self, params, data_loader, model):
        self.params, self.dl, self.model = params, data_loader, model

    def _run(self):
        self.model.train()
        batch = next(iter(self.dl['train']))
        y = batch['y']
        out = self.model(batch)
        if y.dtype.is_floating_point:                      # regression
            assert out.shape == y.shape, (out.shape, y.shape)
            loss = nn.MSELoss()(out, y)
            kind = 'regression'
        else:                                              # classification
            assert out.shape == (y.shape[0], self.params.num_of_classes), out.shape
            loss = nn.CrossEntropyLoss()(out, y)
            kind = 'classification'
        loss.backward()
        n_grad = sum(1 for p in self.model.parameters()
                     if p.grad is not None and p.grad.abs().sum() > 0)
        print(f'[{self.params.downstream_dataset}/{kind}] out={tuple(out.shape)} '
              f'loss={float(loss):.4f} params_with_grad={n_grad}')
        print('CPU_SMOKE_OK')
        return None

    train_for_multiclass = _run
    train_for_regression_dict = _run


finetune_main.Trainer = _CPUSmokeTrainer
finetune_main.main()
