"""
Cache CLIP text embeddings for Alljoined-1.6M images, indexed by image_id.

For each image in the experiment metadata, we embed the class label
("a photo of a {category}") via a CLIP text encoder and store the result
in an H5 file parallel to `pixel_values.h5`, at the same row index.

Usage:
    python scripts/gen_text_embeddings.py \
        --metadata data/Alljoined-1.6M/cache/alljoined/experiment_metadata.parquet \
        --output   data/Alljoined-1.6M/text_embeddings.h5 \
        --model-name openai/clip-vit-large-patch14
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/Alljoined-1.6M/cache/alljoined/experiment_metadata.parquet"),
        help="Parquet with columns image_path and category_name.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/Alljoined-1.6M/text_embeddings.h5"),
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP checkpoint. ViT-L/14 gives 768-d (matches DINOv2-base).",
    )
    p.add_argument(
        "--template",
        type=str,
        default="a photo of a {}",
        help="Prompt template; {} is replaced with the category name.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    return p.parse_args()


def build_id_to_label(metadata_path: Path) -> dict[int, str]:
    df = pd.read_parquet(metadata_path)
    df = df[["image_path", "category_name"]].drop_duplicates()
    df["image_id"] = df["image_path"].str.extract(r"(\d+)\.jpg").astype(int)
    dup = df.groupby("image_id")["category_name"].nunique()
    if (dup > 1).any():
        bad = dup[dup > 1].index.tolist()[:5]
        raise ValueError(f"Conflicting category_name for image_ids {bad}")
    return dict(zip(df["image_id"], df["category_name"]))


def humanize(label) -> str:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""
    return str(label).replace("_", " ").strip()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    id_to_label = build_id_to_label(args.metadata)
    max_id = max(id_to_label)
    n_rows = max_id + 1
    missing_ids = [i for i in range(n_rows) if i not in id_to_label]
    if missing_ids:
        print(f"Warning: {len(missing_ids)} image_ids in [0, {max_id}] have no label; "
              f"their rows will be zero-filled.")

    dtype = np.float16 if args.dtype == "float16" else np.float32

    print(f"Loading CLIP model: {args.model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name)
    model = CLIPTextModelWithProjection.from_pretrained(args.model_name).to(args.device).eval()

    # Probe dim with a single forward
    with torch.inference_mode():
        probe = tokenizer(["probe"], return_tensors="pt", padding=True).to(args.device)
        probe_feat = model(**probe).text_embeds
    embed_dim = probe_feat.shape[-1]
    print(f"Text embedding dim: {embed_dim}  |  rows: {n_rows}  |  dtype: {args.dtype}")

    # Drop image_ids with missing labels; they'll be zero-filled alongside missing_ids.
    id_to_prompt = {}
    n_null_label = 0
    for i, l in id_to_label.items():
        h = humanize(l)
        if not h:
            n_null_label += 1
            continue
        id_to_prompt[i] = args.template.format(h)
    if n_null_label:
        print(f"Warning: {n_null_label} image_ids have null category_name; rows zero-filled.")
    unique_prompts = sorted(set(id_to_prompt.values()))
    prompt_to_idx = {p: i for i, p in enumerate(unique_prompts)}
    print(f"Unique prompts: {len(unique_prompts)} (vs {len(id_to_prompt)} labelled images)")

    unique_feats = np.zeros((len(unique_prompts), embed_dim), dtype=dtype)
    with torch.inference_mode():
        for start in tqdm(range(0, len(unique_prompts), args.batch_size), desc="Encoding"):
            chunk = unique_prompts[start : start + args.batch_size]
            toks = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(args.device)
            feats = model(**toks).text_embeds
            unique_feats[start : start + len(chunk)] = feats.cpu().numpy().astype(dtype, copy=False)

    all_feats = np.zeros((n_rows, embed_dim), dtype=dtype)
    has_text = np.zeros((n_rows,), dtype=bool)
    for image_id, prompt in id_to_prompt.items():
        all_feats[image_id] = unique_feats[prompt_to_idx[prompt]]
        has_text[image_id] = True

    with h5py.File(args.output, "w") as f:
        f.create_dataset("text_embeddings", data=all_feats)
        f.create_dataset("has_text", data=has_text)
        f.attrs["model_name"] = args.model_name
        f.attrs["template"] = args.template
        f.attrs["embed_dim"] = embed_dim

    print(f"Saved {args.output}  ({n_rows} rows × {embed_dim}-d)")


if __name__ == "__main__":
    main()
