"""
Analyze the trained SourceProjector checkpoint for trivial-solution indicators.

Checks:
1. Reconstruction quality (MSE, correlation)
2. Source decorrelation (off-diagonal correlation matrix)
3. Whether mixing weights collapsed (rank, singular values)
4. Whether sources are diverse or trivially identical
5. Per-channel reconstruction quality
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.alignment import SourceProjector
from datasets.cached_dataset import get_webdataset, collate_cached
from torch.utils.data import DataLoader
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='outputs/source_projector_pretrain/epoch20_loss0.017692232504487038.pth')
    parser.add_argument('--num_batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out_dir', type=str, default='figs/source_projector_analysis')
    parser.add_argument('--datasets', type=str, nargs='+', default=['tueg/*.tar', 'Alljoined-1.6M/*.tar'],
                        help='Dataset shard patterns to use')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load model ---
    model = SourceProjector(in_dim=40, num_sources=32, decorr_weight=0.1)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # --- Load data ---
    class FakeParams:
        in_dim = 40
        seq_len = 20
    params = FakeParams()

    dataset = get_webdataset(args.datasets, params)
    dataset = (
        dataset
        .shuffle(1000)
        .batched(args.batch_size, partial=True, collation_fn=collate_cached)
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=4, pin_memory=True)

    # --- Collect stats ---
    all_recon_mse = []
    all_decorr = []
    all_source_stds = []       # std of each source (should vary)
    all_source_means = []
    all_corr_per_channel = []  # per-channel R^2
    all_fwd_weights = []       # mixing weights
    all_source_signals = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            x = batch['timeseries'] / 100  # same scaling as training
            ch_coords = batch['ch_coords']
            valid = batch.get('valid_channel_mask', None)

            # Forward + inverse
            sources = model(x, ch_coords, valid_channel_mask=valid)     # (B, K, N, d)
            recon = model.inverse(sources, ch_coords, valid_channel_mask=valid)  # (B, C, N, d)

            # Valid mask
            vmask = model._valid_channel_mask(ch_coords, valid)  # (B, C)

            # 1. Reconstruction MSE (valid channels only)
            diff = (recon - x) ** 2
            vmask_exp = vmask.float().unsqueeze(-1).unsqueeze(-1)
            mse = (diff * vmask_exp).sum() / (vmask_exp.sum() * x.size(2) * x.size(3))
            all_recon_mse.append(mse.item())

            # 2. Per-channel correlation
            for b_idx in range(x.size(0)):
                for c_idx in range(x.size(1)):
                    if not vmask[b_idx, c_idx]:
                        continue
                    orig = x[b_idx, c_idx].flatten().cpu().numpy()
                    rec = recon[b_idx, c_idx].flatten().cpu().numpy()
                    if orig.std() < 1e-8:
                        continue
                    r = np.corrcoef(orig, rec)[0, 1]
                    all_corr_per_channel.append(r)

            # 3. Source decorrelation
            decorr = model.compute_decorr_loss(sources).item()
            all_decorr.append(decorr)

            # 4. Source diversity: per-source std
            src_flat = sources.view(sources.size(0), sources.size(1), -1)  # (B, K, N*d)
            stds = src_flat.std(dim=-1)  # (B, K)
            means = src_flat.mean(dim=-1)
            all_source_stds.append(stds.cpu().numpy())
            all_source_means.append(means.cpu().numpy())

            # 5. Mixing weights
            coord_pe, _ = model._spherical_positional_encoding(ch_coords)
            w_fwd = model.forward_mix(coord_pe)  # (B, C, K)
            all_fwd_weights.append(w_fwd.cpu().numpy())

            # Save a few source signals for visualization
            if i == 0:
                all_source_signals = sources[0].cpu().numpy()  # (K, N, d)

            print(f"Batch {i+1}/{args.num_batches}: MSE={mse.item():.6f}, decorr={decorr:.6f}")

    # ==================== Analysis ====================
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    # --- 1. Reconstruction ---
    mean_mse = np.mean(all_recon_mse)
    print(f"\n[Reconstruction] Mean MSE: {mean_mse:.6f}")
    corrs = np.array(all_corr_per_channel)
    print(f"[Reconstruction] Per-channel correlation: mean={corrs.mean():.4f}, "
          f"median={np.median(corrs):.4f}, std={corrs.std():.4f}")
    print(f"  Fraction with r > 0.5: {(corrs > 0.5).mean():.2%}")
    print(f"  Fraction with r > 0.8: {(corrs > 0.8).mean():.2%}")
    print(f"  Fraction with r > 0.9: {(corrs > 0.9).mean():.2%}")

    if corrs.mean() < 0.3:
        print("  ** WARNING: Very low reconstruction correlation — model may not have learned useful projections")
    elif corrs.mean() > 0.95:
        print("  ** NOTE: Near-perfect reconstruction — check if sources are just copying inputs")

    # --- 2. Decorrelation ---
    mean_decorr = np.mean(all_decorr)
    print(f"\n[Decorrelation] Mean off-diagonal correlation^2: {mean_decorr:.6f}")
    if mean_decorr < 0.01:
        print("  Sources are well decorrelated (good)")
    elif mean_decorr > 0.1:
        print("  ** WARNING: Sources still highly correlated — decorrelation loss may not be working")

    # --- 3. Source diversity ---
    stds_all = np.concatenate(all_source_stds, axis=0)  # (total_B, K)
    per_source_mean_std = stds_all.mean(axis=0)  # (K,)
    print(f"\n[Source Diversity] Per-source activity (std):")
    print(f"  min={per_source_mean_std.min():.6f}, max={per_source_mean_std.max():.6f}, "
          f"mean={per_source_mean_std.mean():.6f}")
    dead_sources = (per_source_mean_std < 1e-4).sum()
    print(f"  Dead sources (std < 1e-4): {dead_sources}/32")
    if dead_sources > 16:
        print("  ** WARNING: Many dead sources — model collapsed to using few sources")

    # Check if all sources are near-identical
    means_all = np.concatenate(all_source_means, axis=0)
    inter_source_std = per_source_mean_std.std()
    print(f"  Std across source activities: {inter_source_std:.6f}")
    if inter_source_std < 1e-5:
        print("  ** WARNING: All sources have identical activity — TRIVIAL SOLUTION")

    # --- 4. Mixing weight analysis ---
    W = np.concatenate(all_fwd_weights, axis=0)  # (total_B, C, K)
    # Average over batch (same coords may repeat)
    W_mean = W.mean(axis=0)  # (C, K)
    # Check rank via SVD
    U, S, Vh = np.linalg.svd(W_mean, full_matrices=False)
    print(f"\n[Mixing Weights] Singular values of mean forward_mix (top 10):")
    print(f"  {S[:10]}")
    effective_rank = (S > S[0] * 0.01).sum()
    print(f"  Effective rank (sv > 1% of max): {effective_rank}")
    if effective_rank < 3:
        print("  ** WARNING: Very low rank — mixing matrix collapsed")

    # --- 5. Check for identity/trivial projection ---
    # If K >= C, check if W is close to an identity-like matrix
    print(f"\n[Mixing Weights] Weight stats: "
          f"mean={W_mean.mean():.6f}, std={W_mean.std():.6f}, "
          f"abs_mean={np.abs(W_mean).mean():.6f}")

    # ==================== Plots ====================

    # Plot 1: Correlation histogram
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.hist(corrs, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pearson r')
    ax.set_ylabel('Count')
    ax.set_title(f'Per-channel reconstruction correlation\n(mean={corrs.mean():.3f})')
    ax.axvline(corrs.mean(), color='red', linestyle='--', label=f'mean={corrs.mean():.3f}')
    ax.legend()

    # Plot 2: Source activity heatmap
    ax = axes[0, 1]
    im = ax.imshow(stds_all[:min(100, len(stds_all))].T, aspect='auto', cmap='viridis')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Source index')
    ax.set_title('Per-source activity (std) across samples')
    plt.colorbar(im, ax=ax)

    # Plot 3: Per-source mean std bar chart
    ax = axes[0, 2]
    ax.bar(range(32), per_source_mean_std)
    ax.set_xlabel('Source index')
    ax.set_ylabel('Mean std')
    ax.set_title('Average activity per source')

    # Plot 4: Source correlation matrix (from one batch)
    ax = axes[1, 0]
    src_sample = all_source_signals  # (K, N, d)
    src_flat = src_sample.reshape(src_sample.shape[0], -1)
    src_flat = src_flat - src_flat.mean(axis=1, keepdims=True)
    src_flat = src_flat / (src_flat.std(axis=1, keepdims=True) + 1e-8)
    corr_mat = src_flat @ src_flat.T / src_flat.shape[1]
    im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Source correlation matrix (1 sample)')
    ax.set_xlabel('Source')
    ax.set_ylabel('Source')
    plt.colorbar(im, ax=ax)

    # Plot 5: Mixing weight heatmap
    ax = axes[1, 1]
    im = ax.imshow(W_mean, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Source')
    ax.set_ylabel('Channel')
    ax.set_title('Forward mixing weights (mean)')
    plt.colorbar(im, ax=ax)

    # Plot 6: Singular values
    ax = axes[1, 2]
    ax.bar(range(len(S)), S)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular value')
    ax.set_title('SVD of mixing weights')

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, 'analysis.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to {out_path}")

    # Plot 7: Example source signals
    fig2, axes2 = plt.subplots(4, 8, figsize=(24, 8), sharex=True)
    for k in range(32):
        ax = axes2[k // 8, k % 8]
        signal = all_source_signals[k].flatten()
        ax.plot(signal, linewidth=0.5)
        ax.set_title(f'Src {k}', fontsize=8)
        ax.tick_params(labelsize=6)
    plt.suptitle('Source signals (1 sample)', fontsize=14)
    plt.tight_layout()
    out_path2 = os.path.join(args.out_dir, 'source_signals.png')
    plt.savefig(out_path2, dpi=150)
    print(f"Source signals figure saved to {out_path2}")


if __name__ == '__main__':
    main()
