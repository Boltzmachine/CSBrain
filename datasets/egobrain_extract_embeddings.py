"""Offline extractor: pre-compute frozen vision-encoder embeddings of the
EgoBrain frames into per-subject HDF5, so world-model pretraining can load the
embeddings instead of running the encoder on the fly.

In ``sh/pretrain_worldmodel.sh`` the frozen DINOv2 encoder is run ~5x per frame
per step (CLS for image alignment; patch grid for the flip-align lateral
descriptor and the world-model frame objective, each in TWO orientations under
``--frame_averaging``). The encoder is frozen and the frames are deterministic
(read here from the existing uint8 frame cache, already HF resize+center-
cropped), so every embedding is computable once and reused each epoch — removing
the dominant per-step GPU cost.

What is cached, and WHY each is a distinct slice of one forward (verified on
facebook/dinov2-base):
  * ``cls`` / ``cls_flip`` = ``outputs.hidden_states[-1][:, 0]`` — the
    PRE-final-layernorm CLS, exactly what ``CSBrainAlign._dinov2_cls_token``
    consumes for image alignment.
  * ``grid`` / ``grid_flip`` = ``outputs.last_hidden_state[:, 1+n_reg:]`` — the
    POST-final-layernorm patch tokens (CLS + register tokens dropped), exactly
    what ``CSBrainAlign._image_patch_grid`` consumes. ``last_hidden_state ==
    layernorm(hidden_states[-1])``, so pre- and post-LN differ (max-abs ~5 on
    dinov2-base) — they are NOT interchangeable.
The ``_flip`` variants come from re-encoding the HORIZONTALLY-MIRRORED frame
(``torch.flip(pv, dims=[-1])``). Flipping the patch grid in feature space is
NOT a valid substitute (grid cos 0.86; the nb=2 lateral descriptor is sign-
inverted, cos -0.24), so both orientations are stored explicitly.

Layout written (--dtype, default float32; float16 halves the disk):
    <cache_dir>/<subject>.h5
        cls        (n_clips, n_windows, d_img)       <dtype>
        cls_flip   (n_clips, n_windows, d_img)       <dtype>
        grid       (n_clips, n_windows, P, d_img)    <dtype>
        grid_flip  (n_clips, n_windows, P, d_img)    <dtype>
        has_image  (n_clips, n_windows)              bool   (copied from frames)
        attrs: vision_encoder, frame_size, patch_grid_s, n_patches, d_img, dtype,
               n_register_tokens, window_s, stride_s, erp_latency_s, clip_s,
               n_windows, fs_out, format_version

At train time ``EgoBrainDataset`` checks for this dir (when
``--use_cached_embeddings`` is set); if present it reads the embeddings and the
model skips the encoder. The cache invalidates whenever ``window_s``,
``stride_s``, ``erp_latency_s``, ``clip_s``, ``n_windows`` or ``vision_encoder``
change — the dir name encodes them, identically to the frame cache.

Run (formal job on one H100 — see sh/extract_embeddings.sh):
    python -m datasets.egobrain_extract_embeddings \\
        --data_dir data/EgoBrain --subjects all \\
        --vision_encoder facebook/dinov2-base \\
        --window_s 1.0 --stride_s 1.0 --erp_latency_s 0.5 \\
        --n_windows 2 --batch_size 256
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

from datasets.cinebrain_dataset import _encoder_kind, _frame_size_for
from datasets.egobrain_dataset import _get_normalize_params


def frames_cache_slug(vision_encoder: str, window_s: float, stride_s: float,
                      erp_latency_s: float, n_windows: int,
                      frame_size: int) -> str:
    """Dir name of the uint8 FRAME cache (input to this extractor). Mirrors
    ``datasets/egobrain_extract_frames.py`` / ``EgoBrainDataset``."""
    enc_slug = vision_encoder.replace('/', '_')
    return (f'cache_frames_{enc_slug}'
            f'_w{window_s}s{stride_s}_e{erp_latency_s}_nw{n_windows}'
            f'_sz{frame_size}')


def embeddings_cache_slug(vision_encoder: str, window_s: float, stride_s: float,
                          erp_latency_s: float, n_windows: int,
                          frame_size: int) -> str:
    """Dir name of the EMBEDDING cache (output of this extractor). Same
    invalidating fields as the frame cache, ``cache_embeddings_`` prefix."""
    enc_slug = vision_encoder.replace('/', '_')
    return (f'cache_embeddings_{enc_slug}'
            f'_w{window_s}s{stride_s}_e{erp_latency_s}_nw{n_windows}'
            f'_sz{frame_size}')


@torch.no_grad()
def encode_frame_embeddings(model, pixel_values: torch.Tensor,
                            n_register_tokens: Optional[int] = None) -> dict:
    """Run the frozen vision encoder on normalized ``pixel_values`` (B,3,H,W)
    and return the four cached tensors as a dict of CPU float32 tensors.

    This is the single source of truth for the extraction slices — the parity
    test calls it with ``CSBrainAlign.pretrained_image_encoder`` so the cached
    path is byte-checked against the live path.

    * ``cls``      (B, d_img)    = ``hidden_states[-1][:, 0]``         (pre-LN)
    * ``cls_flip`` (B, d_img)    = same, on the horizontally-mirrored frame
    * ``grid``     (B, P, d_img) = ``last_hidden_state[:, 1+n_reg:]``  (post-LN)
    * ``grid_flip``(B, P, d_img) = same, on the mirrored frame
    """
    if n_register_tokens is None:
        n_register_tokens = int(getattr(model.config, 'num_register_tokens', 0) or 0)

    def _one(pv):
        out = model(pixel_values=pv, output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0]                    # (B, d) pre-LN
        grid = out.last_hidden_state[:, 1 + n_register_tokens:, :]  # (B, P, d) post-LN
        return cls.float().cpu(), grid.float().cpu()

    cls, grid = _one(pixel_values)
    cls_flip, grid_flip = _one(torch.flip(pixel_values, dims=[-1]))
    P = grid.size(1)
    s = int(round(math.sqrt(P)))
    assert s * s == P, (
        f"vision patch grid is not square (P={P}); cannot build the "
        f"column-band flip descriptor")
    return {'cls': cls, 'cls_flip': cls_flip, 'grid': grid,
            'grid_flip': grid_flip, 'patch_grid_s': s}


def _extract_subject(sub: str, frames_path: str, out_path: str, model,
                     mean: torch.Tensor, std: torch.Tensor, device: str,
                     batch_size: int, cfg: dict, n_register_tokens: int) -> dict:
    """Read one subject's frame cache, encode every (clip×window) frame in both
    orientations, and write the per-subject embedding HDF5 atomically."""
    if not os.path.exists(frames_path):
        # Expected for the 16 no-video subjects (P0025-P0040); the frame
        # extractor wrote nothing for them, so there is nothing to embed.
        return {'subject': sub, 'status': 'no_frames'}

    with h5py.File(frames_path, 'r') as h:
        frames = np.asarray(h['frames'])          # (n_clips, nw, H, W, 3) uint8
        has_image = np.asarray(h['has_image'])    # (n_clips, nw) bool
        fattrs = dict(h.attrs)
    n_clips, nw, H, Wd, _ = frames.shape

    # Skip a valid existing cache unless --overwrite.
    if os.path.exists(out_path) and not cfg['overwrite']:
        with h5py.File(out_path, 'r') as h:
            a = dict(h.attrs)
        same = (str(a.get('vision_encoder')) == str(cfg['vision_encoder'])
                and int(a.get('frame_size', -1)) == int(cfg['frame_size'])
                and int(a.get('n_windows', -1)) == int(nw)
                and str(a.get('dtype')) == str(cfg['dtype'])
                and int(a.get('format_version', -1)) == int(_FORMAT_VERSION))
        if same:
            return {'subject': sub, 'status': 'skip', 'n_clips': n_clips}

    # Flatten (clip, window) so frames batch uniformly through the encoder.
    np_dtype = np.dtype(cfg['dtype'])
    flat = frames.reshape(n_clips * nw, H, Wd, 3)
    n = flat.shape[0]
    d_img = int(model.config.hidden_size)
    cls_buf = np.zeros((n, d_img), dtype=np_dtype)
    cls_flip_buf = np.zeros((n, d_img), dtype=np_dtype)
    grid_full = None
    grid_flip_full = None
    patch_s = None

    for start in range(0, n, batch_size):
        chunk = flat[start:start + batch_size]
        x = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
        x = x.float().div_(255.0).permute(0, 3, 1, 2)        # (b,3,H,W) on device
        x = (x - mean) / std                                  # ImageNet norm (mean/std on device)
        emb = encode_frame_embeddings(
            model, x, n_register_tokens=n_register_tokens)
        if grid_full is None:
            P = emb['grid'].size(1)
            patch_s = emb['patch_grid_s']
            grid_full = np.zeros((n, P, d_img), dtype=np_dtype)
            grid_flip_full = np.zeros((n, P, d_img), dtype=np_dtype)
        sl = slice(start, start + chunk.shape[0])
        cls_buf[sl] = emb['cls'].numpy().astype(np_dtype)
        cls_flip_buf[sl] = emb['cls_flip'].numpy().astype(np_dtype)
        grid_full[sl] = emb['grid'].numpy().astype(np_dtype)
        grid_flip_full[sl] = emb['grid_flip'].numpy().astype(np_dtype)

    P = grid_full.shape[1]
    cls_buf = cls_buf.reshape(n_clips, nw, d_img)
    cls_flip_buf = cls_flip_buf.reshape(n_clips, nw, d_img)
    grid_full = grid_full.reshape(n_clips, nw, P, d_img)
    grid_flip_full = grid_flip_full.reshape(n_clips, nw, P, d_img)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + '.tmp'
    # Compression is transparent to readers (h5py decompresses automatically), so
    # the choice trades extractor wall-clock against disk. gzip is CPU-bound and
    # barely shrinks high-entropy float32 (~8%); lzf is far faster for ~the same
    # ratio; none is fastest. Default lzf for float32, gzip for float16.
    comp = cfg.get('compression') or ('gzip' if cfg['dtype'] == 'float16' else 'lzf')
    ckw = {'gzip': dict(compression='gzip', compression_opts=4),
           'lzf': dict(compression='lzf'),
           'none': dict()}[comp]
    with h5py.File(tmp_path, 'w') as h:
        dt = cfg['dtype']
        h.create_dataset('cls', data=cls_buf, dtype=dt, chunks=(1, nw, d_img), **ckw)
        h.create_dataset('cls_flip', data=cls_flip_buf, dtype=dt, chunks=(1, nw, d_img), **ckw)
        h.create_dataset('grid', data=grid_full, dtype=dt, chunks=(1, nw, P, d_img), **ckw)
        h.create_dataset('grid_flip', data=grid_flip_full, dtype=dt, chunks=(1, nw, P, d_img), **ckw)
        h.create_dataset('has_image', data=has_image, dtype='bool')
        h.attrs['subject'] = sub
        h.attrs['vision_encoder'] = cfg['vision_encoder']
        h.attrs['frame_size'] = int(cfg['frame_size'])
        h.attrs['patch_grid_s'] = int(patch_s)
        h.attrs['n_patches'] = int(P)
        h.attrs['d_img'] = int(d_img)
        h.attrs['dtype'] = dt
        h.attrs['n_register_tokens'] = int(n_register_tokens)
        # Carry the frame cache's window knobs forward verbatim so the dataset
        # can validate the slug<->content match the same way it does for frames.
        for k in ('window_s', 'stride_s', 'erp_latency_s', 'clip_s',
                  'n_windows', 'fs_out'):
            if k in fattrs:
                h.attrs[k] = fattrs[k]
        h.attrs['format_version'] = _FORMAT_VERSION
    os.replace(tmp_path, out_path)
    return {'subject': sub, 'status': 'ok', 'n_clips': n_clips}


_FORMAT_VERSION = 2


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='data/EgoBrain')
    p.add_argument('--frames_cache_dir', default=None,
                   help='uint8 frame cache to read; default derives the '
                        'cache_frames_<enc>_w.. slug')
    p.add_argument('--cache_dir', default=None,
                   help='embedding cache to write; default derives the '
                        'cache_embeddings_<enc>_w.. slug')
    p.add_argument('--subjects', required=True,
                   help='comma-separated subject ids, or "all"')
    p.add_argument('--vision_encoder', default='facebook/dinov2-base',
                   help='HF model id; must match the frame cache + the run')
    p.add_argument('--frame_size', type=int, default=None,
                   help='override; default derived from vision_encoder')
    p.add_argument('--window_s', type=float, default=1.0)
    p.add_argument('--stride_s', type=float, default=1.0)
    p.add_argument('--erp_latency_s', type=float, default=0.5)
    p.add_argument('--n_windows', type=int, default=2)
    p.add_argument('--fs_out', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=256,
                   help='frames per encoder forward')
    p.add_argument('--dtype', default='float32', choices=['float16', 'float32'],
                   help='on-disk embedding precision. float32 (default) is a '
                        'bit-exact target (~85 GB for dinov2-base); float16 '
                        'halves disk (~46 GB) within the L1/InfoNCE tolerance.')
    p.add_argument('--compression', default=None, choices=['gzip', 'lzf', 'none'],
                   help='HDF5 compressor (transparent to readers). Default lzf for '
                        'float32 / gzip for float16. float32 barely compresses, so '
                        'gzip there is ~2x slower for ~8%% disk — prefer lzf/none.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    if _encoder_kind(args.vision_encoder) == 'vjepa2':
        # V-JEPA 2's alignment rep is an attention pool with a TRAINABLE query,
        # so it is not cacheable the same way; only the frozen grid is. Scope
        # the first pass to DINOv2-style ViTs with a frozen CLS.
        raise SystemExit(
            "egobrain_extract_embeddings currently supports DINOv2-style "
            "encoders only (V-JEPA 2's pooled alignment rep uses a trainable "
            "query and must be recomputed at train time).")

    frame_size = (args.frame_size if args.frame_size is not None
                  else _frame_size_for(args.vision_encoder))

    frames_cache_dir = args.frames_cache_dir or os.path.join(
        args.data_dir, frames_cache_slug(
            args.vision_encoder, args.window_s, args.stride_s,
            args.erp_latency_s, args.n_windows, frame_size))
    cache_dir = args.cache_dir or os.path.join(
        args.data_dir, embeddings_cache_slug(
            args.vision_encoder, args.window_s, args.stride_s,
            args.erp_latency_s, args.n_windows, frame_size))

    if not os.path.isdir(frames_cache_dir):
        raise SystemExit(
            f"frame cache not found at '{frames_cache_dir}'. Build it first "
            f"with datasets.egobrain_extract_frames (same window/stride/erp/"
            f"nw/sz), then re-run this extractor.")

    if args.subjects.lower() == 'all':
        subjects = sorted(
            re.match(r'^(P\d{4})\.h5$', f).group(1)
            for f in os.listdir(frames_cache_dir)
            if re.match(r'^P\d{4}\.h5$', f))
    else:
        subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'embedding {len(subjects)} subject(s) on {device}: '
          f'{frames_cache_dir} -> {cache_dir}')

    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.vision_encoder).eval().to(device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    n_register_tokens = int(getattr(model.config, 'num_register_tokens', 0) or 0)
    mean, std = _get_normalize_params(args.vision_encoder)
    mean, std = mean.to(device), std.to(device)

    cfg = dict(vision_encoder=args.vision_encoder, frame_size=frame_size,
               dtype=args.dtype, compression=args.compression,
               overwrite=args.overwrite)

    results = []
    for sub in tqdm(subjects):
        frames_path = os.path.join(frames_cache_dir, f'{sub}.h5')
        out_path = os.path.join(cache_dir, f'{sub}.h5')
        try:
            r = _extract_subject(sub, frames_path, out_path, model, mean, std,
                                 device, args.batch_size, cfg, n_register_tokens)
        except Exception as e:                   # noqa: BLE001 — report, continue
            r = {'subject': sub, 'status': f'error: {type(e).__name__}: {e}'}
        results.append(r)
        tqdm.write(f"  {sub}: {r['status']}")

    ok = sum(1 for r in results if r['status'] == 'ok')
    skip = sum(1 for r in results if r['status'] == 'skip')
    no_frames = sum(1 for r in results if r['status'] == 'no_frames')
    err = len(results) - ok - skip - no_frames
    for r in results:
        if r['status'].startswith('error'):
            print(f"[err] {r['subject']}: {r['status']}", file=sys.stderr)
    print(f'done: {ok} ok, {skip} skipped, {no_frames} no-frames, {err} errors '
          f'-> {cache_dir}')
    if err:
        sys.exit(1)


if __name__ == '__main__':
    main()
