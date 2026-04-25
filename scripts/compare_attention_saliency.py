"""
Side-by-side comparison of DINO v1 attention rollout and DeepGaze IIE saliency
on a list of images. Models are loaded once; each image produces one PNG with
[input | DINO v1 rollout overlay | DeepGaze IIE overlay].
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
from transformers import AutoImageProcessor, AutoModel

import deepgaze_pytorch


DEFAULT_CENTERBIAS = "outputs/deepgaze_assets/centerbias_mit1003.npy"


def dino_rollout(model, processor, pil_image, device):
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs, output_attentions=True)
    attentions = out.attentions
    seq = attentions[0].shape[-1]
    result = torch.eye(seq, device=device)
    for attn in attentions:
        a = attn[0].mean(dim=0) + torch.eye(seq, device=device)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    cls = result[0, 1:]
    grid = int(round(math.sqrt(cls.shape[0])))

    pv = inputs["pixel_values"][0]
    image_hw = (pv.shape[1], pv.shape[2])
    m = cls.reshape(1, 1, grid, grid)
    m = F.interpolate(m, size=image_hw, mode="bilinear", align_corners=False)
    m = m.squeeze().cpu().float().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)

    mean = torch.tensor(processor.image_mean).view(3, 1, 1).to(device)
    std = torch.tensor(processor.image_std).view(3, 1, 1).to(device)
    disp = (pv * std + mean).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return disp, m


def deepgaze_saliency(model, pil_image, centerbias_template, device):
    image = np.array(pil_image)
    H, W = image.shape[:2]
    cb = zoom(
        centerbias_template,
        (H / centerbias_template.shape[0], W / centerbias_template.shape[1]),
        order=0, mode="nearest",
    )
    cb -= logsumexp(cb)
    img_t = torch.tensor(image.transpose(2, 0, 1)[None], dtype=torch.float32).to(device)
    cb_t = torch.tensor(cb[None], dtype=torch.float32).to(device)
    with torch.inference_mode():
        log_density = model(img_t, cb_t)
    density = torch.exp(log_density).squeeze().cpu().numpy()
    density = (density - density.min()) / (density.max() - density.min() + 1e-8)
    return image, density


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True,
                        help="List of image paths")
    parser.add_argument("--dino-model", type=str, default="facebook/dino-vitb8")
    parser.add_argument("--centerbias", type=str, default=DEFAULT_CENTERBIAS)
    parser.add_argument("--out", type=str, default="outputs/attn_vs_saliency")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.dino_model)
    dino = AutoModel.from_pretrained(
        args.dino_model, attn_implementation="eager"
    ).to(args.device).eval()

    deepgaze = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(args.device).eval()
    cb_template = np.load(args.centerbias)

    for img_path in args.images:
        img_path = Path(img_path)
        pil = Image.open(img_path).convert("RGB")

        disp, dino_map = dino_rollout(dino, processor, pil, args.device)
        orig_image, saliency = deepgaze_saliency(deepgaze, pil, cb_template, args.device)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_image); axes[0].set_title("input"); axes[0].axis("off")
        axes[1].imshow(disp)
        axes[1].imshow(dino_map, cmap="jet", alpha=0.5)
        axes[1].set_title(f"DINO v1 rollout ({args.dino_model.split('/')[-1]})")
        axes[1].axis("off")
        axes[2].imshow(orig_image)
        axes[2].imshow(saliency, cmap="jet", alpha=0.5)
        axes[2].set_title("DeepGaze IIE"); axes[2].axis("off")
        fig.tight_layout()
        out_path = out_dir / f"{img_path.stem}_compare.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
