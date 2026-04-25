"""
Visualize DINOv2 self-attention (CLS → patches) across all layers on a sample image.

The DINOv2 encoder used in models/spectral_alignment.py is loaded exactly as in
the model (facebook/dinov2-base, frozen). For each transformer layer we take the
CLS-token row of the attention matrix, average over heads, reshape to the patch
grid, upsample to image resolution, and overlay on the image.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


DEFAULT_IMAGE = (
    "data/THINGS-EEG2/test_images/00001_aircraft_carrier/aircraft_carrier_06s.jpg"
)


def load_image_tensor(image_path: Path, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return image, inputs


def attention_rollout(attentions, image_hw, discard_ratio: float = 0.0,
                      skip_layers: int = 0):
    """
    Abnar & Zuidema (2020) attention rollout: multiply per-layer attention
    matrices (after accounting for the residual connection) to trace how
    information flows from patches into the CLS token across all layers.

    attentions: tuple of L tensors, each (1, heads, seq, seq).
    Returns a single heatmap of shape image_hw in [0, 1].
    """
    attentions = attentions[skip_layers:]
    seq = attentions[0].shape[-1]
    device = attentions[0].device
    result = torch.eye(seq, device=device)
    for attn in attentions:
        a = attn[0].mean(dim=0)  # (seq, seq), head-averaged
        if discard_ratio > 0:
            flat = a.view(-1)
            k = int(flat.numel() * discard_ratio)
            if k > 0:
                thresh = flat.kthvalue(k).values
                a = torch.where(a < thresh, torch.zeros_like(a), a)
        a = a + torch.eye(seq, device=device)  # residual
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    cls_to_patches = result[0, 1:]  # (Np,)
    grid = int(round(math.sqrt(cls_to_patches.shape[0])))
    m = cls_to_patches.reshape(1, 1, grid, grid)
    m = F.interpolate(m, size=image_hw, mode="bilinear", align_corners=False)
    m = m.squeeze().cpu().float().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m


def extract_cls_attention_maps(attentions, image_hw):
    """
    attentions: tuple of L tensors, each (1, heads, seq, seq)
    Returns a list of L numpy arrays of shape image_hw, each a heatmap in [0,1].
    """
    maps = []
    for attn in attentions:
        cls_to_patches = attn[0, :, 0, 1:]  # (heads, Np)
        a = cls_to_patches.mean(dim=0)  # (Np,)
        np_tokens = a.shape[0]
        grid = int(round(math.sqrt(np_tokens)))
        assert grid * grid == np_tokens, (
            f"Expected square patch grid, got {np_tokens} patches"
        )
        a = a.reshape(1, 1, grid, grid)
        a = F.interpolate(a, size=image_hw, mode="bilinear", align_corners=False)
        a = a.squeeze().cpu().float().numpy()
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        maps.append(a)
    return maps


def save_grid(image_np, maps, out_path: Path):
    n = len(maps)
    cols = 4
    rows = math.ceil((n + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(image_np)
    axes[0].set_title("input")
    axes[0].axis("off")

    for i, m in enumerate(maps):
        ax = axes[i + 1]
        ax.imshow(image_np)
        ax.imshow(m, cmap="jet", alpha=0.5)
        ax.set_title(f"layer {i}")
        ax.axis("off")

    for ax in axes[n + 1:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_image(image_path: Path, out_dir: Path, model, processor, device,
                  save_per_layer: bool, rollout_skip_layers: int = 0):
    image, inputs = load_image_tensor(image_path, processor)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True)

    pixel_values = inputs["pixel_values"][0]
    image_hw = (pixel_values.shape[1], pixel_values.shape[2])

    mean = torch.tensor(processor.image_mean).view(3, 1, 1).to(pixel_values.device)
    std = torch.tensor(processor.image_std).view(3, 1, 1).to(pixel_values.device)
    disp = (pixel_values * std + mean).clamp(0, 1).cpu().permute(1, 2, 0).numpy()

    maps = extract_cls_attention_maps(outputs.attentions, image_hw)
    save_grid(disp, maps, out_dir / f"{image_path.stem}_all_layers.png")

    rollout = attention_rollout(outputs.attentions, image_hw,
                                 skip_layers=rollout_skip_layers)
    total = len(outputs.attentions)
    rollout_title = (
        f"attention rollout (layers {rollout_skip_layers}..{total - 1})"
        if rollout_skip_layers else "attention rollout (all layers)"
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(disp); axes[0].set_title("input"); axes[0].axis("off")
    axes[1].imshow(disp)
    axes[1].imshow(rollout, cmap="jet", alpha=0.5)
    axes[1].set_title(rollout_title)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{image_path.stem}_rollout.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    if save_per_layer:
        for i, m in enumerate(maps):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(disp); ax.imshow(m, cmap="jet", alpha=0.5)
            ax.set_title(f"layer {i}"); ax.axis("off")
            fig.savefig(out_dir / f"{image_path.stem}_layer{i:02d}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    return len(maps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path (ignored if --images is set)")
    parser.add_argument("--images", nargs="+", default=None,
                        help="List of image paths (batch mode)")
    parser.add_argument("--model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--out", type=str, default="outputs/dino_attention")
    parser.add_argument("--no-per-layer", action="store_true",
                        help="Skip per-layer PNGs, only save grid + rollout")
    parser.add_argument("--rollout-skip-layers", type=int, default=0,
                        help="Exclude the first N layers when computing rollout")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.images:
        image_paths = [Path(p) for p in args.images]
    else:
        image_paths = [Path(args.image or DEFAULT_IMAGE)]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.model)
    # SDPA doesn't expose attention weights; eager does.
    model = AutoModel.from_pretrained(
        args.model, attn_implementation="eager"
    ).to(args.device).eval()

    for p in image_paths:
        n = process_image(p, out_dir, model, processor, args.device,
                          save_per_layer=not args.no_per_layer,
                          rollout_skip_layers=args.rollout_skip_layers)
        print(f"  [{p.name}] {n} layers → {out_dir}")


if __name__ == "__main__":
    main()
