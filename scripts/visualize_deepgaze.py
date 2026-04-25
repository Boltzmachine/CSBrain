"""
Run DeepGaze IIE (human fixation prediction) on a sample image and save a
saliency overlay, for comparison with the DINO attention maps produced by
scripts/visualize_dino_attention.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp

import deepgaze_pytorch


DEFAULT_IMAGE = (
    "data/THINGS-EEG2/test_images/00001_aircraft_carrier/aircraft_carrier_06s.jpg"
)
DEFAULT_CENTERBIAS = "outputs/deepgaze_assets/centerbias_mit1003.npy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--centerbias", type=str, default=DEFAULT_CENTERBIAS)
    parser.add_argument("--out", type=str, default="outputs/deepgaze_iie")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(args.device).eval()

    image = np.array(Image.open(image_path).convert("RGB"))
    H, W = image.shape[:2]

    centerbias_template = np.load(args.centerbias)
    centerbias = zoom(
        centerbias_template,
        (H / centerbias_template.shape[0], W / centerbias_template.shape[1]),
        order=0, mode="nearest",
    )
    centerbias -= logsumexp(centerbias)

    image_tensor = torch.tensor(image.transpose(2, 0, 1)[None], dtype=torch.float32).to(args.device)
    centerbias_tensor = torch.tensor(centerbias[None], dtype=torch.float32).to(args.device)

    with torch.inference_mode():
        log_density = model(image_tensor, centerbias_tensor)

    density = torch.exp(log_density).squeeze().cpu().numpy()
    density_norm = (density - density.min()) / (density.max() - density.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image); axes[0].set_title("input"); axes[0].axis("off")
    axes[1].imshow(density_norm, cmap="jet")
    axes[1].set_title("DeepGaze IIE saliency"); axes[1].axis("off")
    axes[2].imshow(image)
    axes[2].imshow(density_norm, cmap="jet", alpha=0.5)
    axes[2].set_title("overlay"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{image_path.stem}_deepgaze.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved DeepGaze IIE saliency → {out_dir / (image_path.stem + '_deepgaze.png')}")


if __name__ == "__main__":
    main()
