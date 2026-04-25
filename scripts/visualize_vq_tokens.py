"""
Visualize the Llama-2 vocabulary tokens that the LLM-embedding VQ actually
selects during pre-training.

The VQ codebook contains the top-N (default 4096) most-frequent Llama-2 tokens.
Each forward pass quantizes K = num_language_tokens global slots to codebook
entries; we collect those indices across many samples and map them back to
token strings via the Llama-2 tokenizer.

Outputs:
  outputs/vq_tokens/<run>/top_tokens.png   — bar chart of top-50 tokens
  outputs/vq_tokens/<run>/token_counts.json — full {token_str: count} table
  outputs/vq_tokens/<run>/summary.txt      — utilization, top-50 printout
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets.cached_dataset import get_webdataset, collate_cached
from models import get_model


def load_checkpoint_params(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'state_dict' not in ckpt:
        raise ValueError(f"Expected dict with 'state_dict' key, got {type(ckpt)}")
    params_dict = ckpt['params']
    state_dict = ckpt['state_dict']
    return params_dict, state_dict


def build_model_from_params(params_dict):
    # Reconstruct Namespace for get_model.
    class Params: pass
    params = Params()
    for k, v in params_dict.items():
        setattr(params, k, v)
    # get_model currently needs brain_regions and sorted_indices as args but
    # CSBrainAlign ignores brain_regions beyond passing it along.
    model = get_model(params, brain_regions=None, sorted_indices=[])
    return model, params


@torch.no_grad()
def run_collection(model, data_loader, device, n_batches, tokenizer):
    """Run forward passes, collect VQ indices via a hook on llm_vq.

    Returns Counter keyed by (llama_token_id, token_str).
    """
    counter = Counter()

    captured = {'indices': None}
    orig_forward = model.llm_vq.forward

    def hook_forward(x):
        out, info = orig_forward(x)
        captured['indices'] = info['indices'].detach().cpu()
        return out, info

    model.llm_vq.forward = hook_forward

    # Llama-ID lookup (codebook idx -> llama token id)
    selected_token_ids = model.llm_vq.selected_token_ids.cpu().tolist()

    model.eval()
    it = iter(data_loader)
    seen = 0
    for step in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        # Only keep samples that carry images (VQ branch requires has_image).
        has_image = batch.get('has_image')
        if has_image is None or not has_image.any():
            continue

        batch_dev = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_dev[k] = v.to(device)
            elif isinstance(v, dict):
                batch_dev[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                                for kk, vv in v.items()}
            else:
                batch_dev[k] = v
        batch_dev['timeseries'] = batch_dev['timeseries'] / 100

        try:
            model(batch_dev, mask=None)
        except Exception as e:
            print(f"[step {step}] forward failed: {e}")
            continue

        idx = captured['indices']
        if idx is None:
            continue
        # idx shape: (B_with_img, K). But only samples with has_image go through
        # the VQ branch, since the forward does branch_embs = patch_emb[has_image].
        idx_flat = idx.reshape(-1).tolist()
        for vq_idx in idx_flat:
            tok_id = selected_token_ids[vq_idx]
            counter[tok_id] += 1
        seen += idx.numel()
        if step % 10 == 0:
            print(f"[step {step}] collected {seen} token selections, "
                  f"unique codebook entries so far: {len(counter)}")
        captured['indices'] = None

    return counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/alljoined-vq/epoch27_loss4.7894816398620605.pth")
    parser.add_argument("--n_batches", type=int, default=50,
                        help="number of forward passes to run")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="default: outputs/vq_tokens/<run_name>")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top_k_plot", type=int, default=50)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    run_name = ckpt_path.parent.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/vq_tokens") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    params_dict, state_dict = load_checkpoint_params(ckpt_path)
    print(f"Saved params keys: {len(params_dict)}  "
          f"use_llm_vq={params_dict.get('use_llm_vq')}  "
          f"num_language_tokens={params_dict.get('num_language_tokens')}  "
          f"codebook_size={params_dict.get('max_llm_codebook_size')}")

    assert params_dict.get('use_llm_vq'), "Checkpoint was not trained with use_llm_vq=True"

    # --- Build model & load weights ---
    model, params = build_model_from_params(params_dict)

    # Non-persistent buffers (selected_token_ids, llm_embedding_bank) are
    # regenerated in set_llm_embedding_bank at build time; the checkpoint only
    # stores learnable + persistent buffers. Missing/unexpected keys are
    # tolerated.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  sample missing:", missing[:5])
    if unexpected:
        print("  sample unexpected:", unexpected[:5])
    model.to(args.device).eval()

    # --- Build dataloader: Alljoined shards only, so every sample has images. ---
    # Minimal params stub for get_webdataset. The function reads
    # params.in_dim and params.seq_len for the crop path.
    if not hasattr(params, 'in_dim'):
        params.in_dim = params_dict.get('in_dim', 40)
    if not hasattr(params, 'seq_len'):
        params.seq_len = params_dict.get('seq_len', 20)

    dataset = get_webdataset(["Alljoined-1.6M/*.tar"], params)
    dataset = (
        dataset
        .shuffle(2000)
        .batched(args.batch_size, partial=True, collation_fn=collate_cached)
    )
    loader = DataLoader(
        dataset, batch_size=None, num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Llama-2 tokenizer for pretty printing ---
    print("Loading Llama-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # --- Collect ---
    counter = run_collection(model, loader, args.device, args.n_batches, tokenizer)

    # --- Map counts to strings ---
    total = sum(counter.values())
    str_counts = []
    for tok_id, c in counter.items():
        tok_str = tokenizer.decode([tok_id])
        # Also include the raw token piece (pre-decode) to preserve leading ▁ etc.
        raw = tokenizer.convert_ids_to_tokens(tok_id)
        str_counts.append((tok_id, raw, tok_str, c))
    str_counts.sort(key=lambda t: -t[3])

    codebook_size = params_dict.get('max_llm_codebook_size', 4096)
    unique_used = len(counter)
    utilization = unique_used / codebook_size

    # --- JSON dump (full distribution) ---
    json_path = out_dir / "token_counts.json"
    json.dump(
        [{"token_id": tid, "raw": raw, "decoded": dec, "count": c,
          "frac": c / total}
         for tid, raw, dec, c in str_counts],
        open(json_path, "w"), ensure_ascii=False, indent=2,
    )

    # --- Text summary ---
    top_k = args.top_k_plot
    summary_lines = [
        f"Checkpoint: {ckpt_path}",
        f"Total token selections: {total}",
        f"Codebook size:          {codebook_size}",
        f"Unique codes used:      {unique_used}  "
        f"({100 * utilization:.2f}%)",
        "",
        f"Top {top_k} tokens:",
        f"{'rank':>4}  {'id':>6}  {'raw':<20}  {'decoded':<20}  {'count':>8}  {'frac':>7}",
    ]
    for i, (tid, raw, dec, c) in enumerate(str_counts[:top_k]):
        summary_lines.append(
            f"{i+1:>4}  {tid:>6}  {raw:<20}  {repr(dec):<20}  "
            f"{c:>8}  {100 * c / total:6.2f}%"
        )
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary)

    # --- Bar chart of top-K ---
    top = str_counts[:top_k]
    labels = [raw for _, raw, _, _ in top]
    counts = [c for _, _, _, c in top]
    fig, ax = plt.subplots(figsize=(max(8, top_k * 0.25), 5))
    ax.bar(range(len(top)), counts)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=8)
    ax.set_ylabel("# selections")
    ax.set_title(f"Top {top_k} VQ tokens ({unique_used}/{codebook_size} codes used)")
    fig.tight_layout()
    fig.savefig(out_dir / "top_tokens.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Log-rank plot: how skewed is the usage? ---
    ranks = np.arange(1, unique_used + 1)
    vals = np.array([c for _, _, _, c in str_counts])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(ranks, vals)
    ax.set_xlabel("rank")
    ax.set_ylabel("count")
    ax.set_title("VQ token usage rank-frequency")
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rank_frequency.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved outputs to {out_dir}")


if __name__ == '__main__':
    main()
