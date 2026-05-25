"""Per-parameter gradient + weight-drift analyzer for finetuning.

Records two complementary signals for every trainable parameter:

  - Instantaneous *gradient norm* at each optimizer step (mean / rms / max
    across the run).
  - Cumulative *weight drift* ``||theta_t - theta_0|| / (||theta_0|| + eps)``
    snapshotted at the end of every epoch.

The two together tell different stories: a layer can have small grads but
still drift far (high effective LR + many steps), or huge grads that average
out to near-zero drift (cancelling updates). Plotting both side-by-side is
usually what you want to spot "barely touched" vs "rewritten" components.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List

import torch
from torch.nn.parameter import UninitializedParameter


def _is_uninitialized(p: torch.nn.Parameter) -> bool:
    return isinstance(p, UninitializedParameter)


def _component_of(name: str) -> str:
    """Coarse grouping prefix used for the per-component summary.

    For ``backbone.encoder.layers.7.attn.qkv.weight`` we group at
    ``backbone.encoder.layers.7`` so each transformer block becomes one
    bucket; everything else uses its first two dotted segments.
    """
    parts = name.split('.')
    if len(parts) >= 4 and parts[1] == 'encoder' and parts[2] == 'layers':
        return '.'.join(parts[:4])
    if len(parts) >= 2:
        return '.'.join(parts[:2])
    return name


class GradientAnalyzer:
    def __init__(self, model: torch.nn.Module, out_dir: str):
        self.model = model
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.init_snapshot: Dict[str, torch.Tensor] = {}
        self.init_norm: Dict[str, float] = {}
        self.grad_sum_sq: Dict[str, float] = defaultdict(float)
        self.grad_max: Dict[str, float] = defaultdict(float)
        self.grad_count: Dict[str, int] = defaultdict(int)

        # rec keys are param names; values are dicts {'abs': .., 'rel': ..}
        self.epoch_drifts: List[Dict] = []

        self._snapshot_initial()

    def _snapshot_initial(self):
        # Lazy-initialised params (e.g. ``nn.LazyLinear``) have no shape until
        # the first forward pass — we'll capture them on first ``after_backward``.
        for name, param in self.model.named_parameters():
            if param.requires_grad and not _is_uninitialized(param):
                self.init_snapshot[name] = param.detach().clone().cpu()
                self.init_norm[name] = float(param.detach().norm().item())

    @torch.no_grad()
    def after_backward(self):
        """Call AFTER ``loss.backward()`` and BEFORE ``optimizer.step()``."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if name not in self.init_snapshot and not _is_uninitialized(param):
                # Newly-materialised lazy param: snapshot now so drift is
                # measured relative to its first concrete weights.
                self.init_snapshot[name] = param.detach().clone().cpu()
                self.init_norm[name] = float(param.detach().norm().item())
            g = float(param.grad.detach().norm().item())
            self.grad_sum_sq[name] += g * g
            if g > self.grad_max[name]:
                self.grad_max[name] = g
            self.grad_count[name] += 1

    @torch.no_grad()
    def end_of_epoch(self, epoch: int):
        rec = {'_epoch': epoch}
        for name, param in self.model.named_parameters():
            if not param.requires_grad or name not in self.init_snapshot:
                continue
            init = self.init_snapshot[name].to(param.device)
            delta = float((param.detach() - init).norm().item())
            init_n = self.init_norm[name]
            rec[name] = {'abs': delta, 'rel': delta / (init_n + 1e-12)}
        self.epoch_drifts.append(rec)

    def finalize(self) -> Dict[str, str]:
        per_param_csv = os.path.join(self.out_dir, 'per_param_stats.csv')
        per_epoch_csv = os.path.join(self.out_dir, 'per_epoch_drift.csv')
        per_comp_csv = os.path.join(self.out_dir, 'per_component_summary.csv')

        last = self.epoch_drifts[-1] if self.epoch_drifts else {}

        # ---- per-parameter ----
        rows = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad or _is_uninitialized(param):
                continue
            n = self.grad_count.get(name, 0)
            ssq = self.grad_sum_sq.get(name, 0.0)
            rms = (ssq / n) ** 0.5 if n > 0 else 0.0
            d = last.get(name, {'abs': 0.0, 'rel': 0.0})
            rows.append({
                'name': name,
                'component': _component_of(name),
                'numel': int(param.numel()),
                'init_norm': self.init_norm.get(name, 0.0),
                'rms_grad_norm': rms,
                'max_grad_norm': self.grad_max.get(name, 0.0),
                'final_abs_drift': d['abs'],
                'final_rel_drift': d['rel'],
            })

        with open(per_param_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # ---- per-epoch relative drift ----
        if self.epoch_drifts:
            names = [k for k in self.epoch_drifts[0].keys() if k != '_epoch']
            with open(per_epoch_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['epoch'] + names)
                for rec in self.epoch_drifts:
                    w.writerow([rec['_epoch']] + [rec[n]['rel'] for n in names])

        # ---- per-component aggregation ----
        comp_rows: Dict[str, Dict] = {}
        for r in rows:
            c = r['component']
            agg = comp_rows.setdefault(c, {
                'component': c, 'n_params': 0, 'numel': 0,
                'rms_grad_norm_weighted': 0.0,  # weighted by numel
                'final_rel_drift_weighted': 0.0,
                'max_grad_norm': 0.0,
            })
            agg['n_params'] += 1
            agg['numel'] += r['numel']
            agg['rms_grad_norm_weighted'] += r['rms_grad_norm'] * r['numel']
            agg['final_rel_drift_weighted'] += r['final_rel_drift'] * r['numel']
            if r['max_grad_norm'] > agg['max_grad_norm']:
                agg['max_grad_norm'] = r['max_grad_norm']
        for agg in comp_rows.values():
            if agg['numel']:
                agg['rms_grad_norm'] = agg['rms_grad_norm_weighted'] / agg['numel']
                agg['final_rel_drift'] = agg['final_rel_drift_weighted'] / agg['numel']
            else:
                agg['rms_grad_norm'] = 0.0
                agg['final_rel_drift'] = 0.0
            del agg['rms_grad_norm_weighted']
            del agg['final_rel_drift_weighted']

        comp_sorted = sorted(comp_rows.values(),
                             key=lambda x: x['final_rel_drift'], reverse=True)
        with open(per_comp_csv, 'w', newline='') as f:
            fns = ['component', 'n_params', 'numel',
                   'rms_grad_norm', 'max_grad_norm', 'final_rel_drift']
            w = csv.DictWriter(f, fieldnames=fns)
            w.writeheader()
            for r in comp_sorted:
                w.writerow({k: r[k] for k in fns})

        try:
            self._plot(rows, comp_sorted)
        except Exception as exc:  # plotting must never break the run
            print(f"[GradAnalyzer] plot failed: {exc}")

        return {
            'per_param_csv': per_param_csv,
            'per_epoch_csv': per_epoch_csv,
            'per_component_csv': per_comp_csv,
        }

    def _plot(self, rows: List[Dict], comp_sorted: List[Dict]):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        K = min(30, len(rows))
        rows_by_drift = sorted(rows, key=lambda r: r['final_rel_drift'], reverse=True)
        top = rows_by_drift[:K]
        bot = list(reversed(rows_by_drift[-K:]))

        # 1) top/bottom relative drift bar chart
        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, K * 0.3)))
        axes[0].barh([r['name'] for r in top][::-1],
                     [r['final_rel_drift'] for r in top][::-1],
                     color='tab:red')
        axes[0].set_xlabel('relative drift ||Δθ|| / ||θ₀||')
        axes[0].set_title(f'Top {K} most-moved parameters')
        axes[1].barh([r['name'] for r in bot][::-1],
                     [r['final_rel_drift'] for r in bot][::-1],
                     color='tab:blue')
        axes[1].set_xlabel('relative drift ||Δθ|| / ||θ₀||')
        axes[1].set_title(f'Bottom {K} least-moved parameters')
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'rel_drift_top_bottom.png'), dpi=120)
        plt.close(fig)

        # 2) per-component bar charts (drift + grad rms, sorted by drift)
        comps = comp_sorted
        labels = [c['component'] for c in comps]
        drifts = [c['final_rel_drift'] for c in comps]
        grads = [c['rms_grad_norm'] for c in comps]
        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(comps) * 0.3)))
        axes[0].barh(labels[::-1], drifts[::-1], color='tab:red')
        axes[0].set_xlabel('avg relative drift (numel-weighted)')
        axes[0].set_title('Per-component cumulative weight drift')
        axes[1].barh(labels[::-1], grads[::-1], color='tab:purple')
        axes[1].set_xlabel('avg RMS grad norm (numel-weighted)')
        axes[1].set_title('Per-component avg gradient magnitude')
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'per_component.png'), dpi=120)
        plt.close(fig)

        # 3) drift vs grad scatter, log-log, every parameter
        xs = np.array([max(r['rms_grad_norm'], 1e-12) for r in rows])
        ys = np.array([max(r['final_rel_drift'], 1e-12) for r in rows])
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(xs, ys, s=8, alpha=0.6)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('RMS grad norm (log)')
        ax.set_ylabel('relative drift (log)')
        ax.set_title('Per-parameter: gradient magnitude vs cumulative drift')
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'grad_vs_drift_scatter.png'), dpi=120)
        plt.close(fig)

        # 4) drift trajectory across epochs for top-K
        if self.epoch_drifts:
            top_names = [r['name'] for r in rows_by_drift[:min(15, len(rows_by_drift))]]
            fig, ax = plt.subplots(figsize=(10, 6))
            for n in top_names:
                ys = [rec[n]['rel'] for rec in self.epoch_drifts]
                xs = [rec['_epoch'] for rec in self.epoch_drifts]
                ax.plot(xs, ys, marker='o', label=n)
            ax.set_xlabel('epoch')
            ax.set_ylabel('relative drift')
            ax.set_title('Drift trajectory: top-moved parameters')
            ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.02, 1.0))
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, 'drift_trajectory_top.png'),
                        dpi=120, bbox_inches='tight')
            plt.close(fig)
