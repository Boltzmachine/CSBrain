"""Offline extractor: pre-compute frozen DINOv2 embeddings of the CONTINUOUS,
time-keyed EgoBrain frame grid into per-subject HDF5.

This is the time-keyed counterpart of :mod:`datasets.egobrain_extract_embeddings`
(which is clip-keyed). It reads the grid frame cache written by
:mod:`datasets.egobrain_extract_frames_grid` (frames keyed by absolute EEG-clock
time, slot k = k*grid_s) and stores the SAME four slices per slot that the
clip-keyed cache stores per (clip, window):

  * ``cls`` / ``cls_flip``   (n_slots, d_img)    = ``hidden_states[-1][:, 0]``
  * ``grid`` / ``grid_flip`` (n_slots, P, d_img) = ``last_hidden_state[:, 1+nreg:]``

so ``EgoBrainDataset(use_frame_grid=True, use_grid_embeddings=True)`` can fetch
the embedding at slot = round((window_centre + erp)/grid_s) and skip the live
encoder — getting BOTH the encoder-skip speedup AND the continuous-offset
variety. The slices are produced by the shared ``encode_frame_embeddings`` (the
single source of truth the clip-keyed extractor + parity test use), so the
embeddings are byte-identical to the clip-keyed cache for the same frame.

Layout written (format_version 2: orientations INTERLEAVED on axis 1 — [:,0]=
orig, [:,1]=horizontal-flip — so one per-slot chunk read returns both, halving
the expensive grid seeks per training item with no read amplification):
    <cache_dir>/<subject>.h5
        cls        (n_slots, 2, d_img)     <dtype>   chunks (1,2,d_img)
        grid       (n_slots, 2, P, d_img)  <dtype>   chunks (1,2,P,d_img)
        has_image  (n_slots,)              bool   (copied from the frame grid)
        attrs: vision_encoder, frame_size, grid_s, patch_grid_s, n_patches,
               d_img, dtype, n_register_tokens, n_slots, orient_axis, format_version

Dir name mirrors the frame grid (only encoder/frame_size/grid_s change bytes):
    cache_embeddings_grid_<enc>_g<grid_s>_sz<frame_size>/

DINOv2-style encoders only (same constraint as the clip-keyed extractor:
V-JEPA 2's pooled alignment rep uses a trainable query, not cacheable).

Run (see sh/extract_embeddings_grid.sh — one H100):
    python -m datasets.egobrain_extract_embeddings_grid \\
        --data_dir data/EgoBrain --subjects all \\
        --vision_encoder facebook/dinov2-base --grid_s 0.2 \\
        --dtype float16 --batch_size 256
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

from datasets.cinebrain_dataset import _encoder_kind, _frame_size_for
from datasets.egobrain_dataset import _get_normalize_params
from datasets.egobrain_extract_embeddings import encode_frame_embeddings
from datasets.egobrain_extract_frames_grid import _grid_cache_dir


_FORMAT_VERSION = 2   # v2: orientations interleaved -> cls (n,2,d), grid (n,2,P,d)


def emb_grid_cache_dir(data_dir: str, vision_encoder: str, grid_s: float,
                       frame_size: int) -> str:
    """Dir name of the time-keyed embedding cache. Same byte-changing knobs as
    the frame grid (encoder -> resize/crop + frame_size, and grid_s)."""
    enc_slug = vision_encoder.replace('/', '_')
    return os.path.join(
        data_dir,
        f'cache_embeddings_grid_{enc_slug}_g{grid_s}_sz{frame_size}')


def _extract_subject(sub: str, frames_path: str, out_path: str, model,
                     mean: torch.Tensor, std: torch.Tensor, device: str,
                     batch_size: int, cfg: dict, n_register_tokens: int) -> dict:
    """Encode one subject's grid frames (n_slots,H,W,3) into the four embedding
    slices (n_slots,...) and write the per-subject HDF5 atomically."""
    if not os.path.exists(frames_path):
        return {'subject': sub, 'status': 'no_frames'}

    with h5py.File(frames_path, 'r') as h:
        frames = np.asarray(h['frames'])            # (n_slots, H, W, 3) uint8
        has_image = np.asarray(h['has_image'])      # (n_slots,) bool
        fattrs = dict(h.attrs)
    n_slots, H, Wd, _ = frames.shape

    if os.path.exists(out_path) and not cfg['overwrite']:
        with h5py.File(out_path, 'r') as h:
            a = dict(h.attrs)
        same = (str(a.get('vision_encoder')) == str(cfg['vision_encoder'])
                and int(a.get('frame_size', -1)) == int(cfg['frame_size'])
                and a.get('grid_s') is not None
                and abs(float(a['grid_s']) - float(cfg['grid_s'])) < 1e-9
                and int(a.get('n_slots', -1)) == int(n_slots)
                and str(a.get('dtype')) == str(cfg['dtype'])
                and int(a.get('format_version', -1)) == int(_FORMAT_VERSION))
        if same:
            return {'subject': sub, 'status': 'skip', 'n_slots': n_slots}

    np_dtype = np.dtype(cfg['dtype'])
    d_img = int(model.config.hidden_size)
    # Orientations interleaved along axis 1 ([:, 0] = orig, [:, 1] = h-flip) so
    # a single per-slot chunk read returns BOTH orientations -> halves the
    # (expensive) grid seeks per training item, with zero read amplification
    # (both orientations are always used together by the model).
    cls_buf = np.zeros((n_slots, 2, d_img), dtype=np_dtype)
    grid_full = None                                    # -> (n_slots, 2, P, d)
    patch_s = None

    for start in range(0, n_slots, batch_size):
        chunk = frames[start:start + batch_size]
        x = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
        x = x.float().div_(255.0).permute(0, 3, 1, 2)
        x = (x - mean) / std
        emb = encode_frame_embeddings(
            model, x, n_register_tokens=n_register_tokens)
        if grid_full is None:
            P = emb['grid'].size(1)
            patch_s = emb['patch_grid_s']
            grid_full = np.zeros((n_slots, 2, P, d_img), dtype=np_dtype)
        sl = slice(start, start + chunk.shape[0])
        cls_buf[sl, 0] = emb['cls'].numpy().astype(np_dtype)
        cls_buf[sl, 1] = emb['cls_flip'].numpy().astype(np_dtype)
        grid_full[sl, 0] = emb['grid'].numpy().astype(np_dtype)
        grid_full[sl, 1] = emb['grid_flip'].numpy().astype(np_dtype)

    P = grid_full.shape[2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + '.tmp'
    comp = cfg.get('compression') or ('gzip' if cfg['dtype'] == 'float16' else 'lzf')
    ckw = {'gzip': dict(compression='gzip', compression_opts=4),
           'lzf': dict(compression='lzf'),
           'none': dict()}[comp]
    with h5py.File(tmp_path, 'w') as h:
        dt = cfg['dtype']
        # Per-slot chunks over the interleaved arrays: one chunk = one slot's
        # BOTH orientations, so train-time reads fetch a window's cls/grid in a
        # single seek each.
        h.create_dataset('cls', data=cls_buf, dtype=dt, chunks=(1, 2, d_img), **ckw)
        h.create_dataset('grid', data=grid_full, dtype=dt,
                         chunks=(1, 2, P, d_img), **ckw)
        h.create_dataset('has_image', data=has_image, dtype='bool')
        h.attrs['subject'] = sub
        h.attrs['vision_encoder'] = cfg['vision_encoder']
        h.attrs['frame_size'] = int(cfg['frame_size'])
        h.attrs['grid_s'] = float(cfg['grid_s'])
        h.attrs['patch_grid_s'] = int(patch_s)
        h.attrs['orient_axis'] = 1            # axis 1: [0]=orig, [1]=h-flip
        h.attrs['n_patches'] = int(P)
        h.attrs['d_img'] = int(d_img)
        h.attrs['dtype'] = dt
        h.attrs['n_register_tokens'] = int(n_register_tokens)
        h.attrs['n_slots'] = int(n_slots)
        for k in ('clip_s', 'fs_out', 'video_offset_s', 'total_duration_s'):
            if k in fattrs:
                h.attrs[k] = fattrs[k]
        h.attrs['format_version'] = _FORMAT_VERSION
    os.replace(tmp_path, out_path)
    return {'subject': sub, 'status': 'ok', 'n_slots': n_slots}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='data/EgoBrain')
    p.add_argument('--frames_cache_dir', default=None,
                   help='grid frame cache to read; default derives the '
                        'cache_frames_grid_<enc>_g<g>_sz<sz> slug')
    p.add_argument('--cache_dir', default=None,
                   help='embedding cache to write; default derives the '
                        'cache_embeddings_grid_<enc>_g<g>_sz<sz> slug')
    p.add_argument('--subjects', required=True,
                   help='comma-separated subject ids, or "all"')
    p.add_argument('--vision_encoder', default='facebook/dinov2-base',
                   help='HF model id; must match the frame grid + the run')
    p.add_argument('--frame_size', type=int, default=None,
                   help='override; default derived from vision_encoder')
    p.add_argument('--grid_s', type=float, default=0.2,
                   help='time grid spacing in seconds (must match the frame grid)')
    p.add_argument('--batch_size', type=int, default=256,
                   help='frames per encoder forward')
    p.add_argument('--dtype', default='float16', choices=['float16', 'float32'],
                   help='on-disk precision. float16 (default) halves disk and '
                        'is within the L1/InfoNCE tolerance (the dataset upcasts '
                        'to float32 on read); float32 is bit-exact but ~2x disk.')
    p.add_argument('--compression', default=None, choices=['gzip', 'lzf', 'none'],
                   help='HDF5 compressor (transparent to readers). Default gzip '
                        'for float16 / lzf for float32.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    if _encoder_kind(args.vision_encoder) == 'vjepa2':
        raise SystemExit(
            "egobrain_extract_embeddings_grid supports DINOv2-style encoders "
            "only (V-JEPA 2's pooled alignment rep uses a trainable query).")

    frame_size = (args.frame_size if args.frame_size is not None
                  else _frame_size_for(args.vision_encoder))
    frames_cache_dir = args.frames_cache_dir or _grid_cache_dir(
        args.data_dir, args.vision_encoder, args.grid_s, frame_size)
    cache_dir = args.cache_dir or emb_grid_cache_dir(
        args.data_dir, args.vision_encoder, args.grid_s, frame_size)

    if not os.path.isdir(frames_cache_dir):
        raise SystemExit(
            f"grid frame cache not found at '{frames_cache_dir}'. Build it "
            f"first with datasets.egobrain_extract_frames_grid (same "
            f"vision_encoder/grid_s), then re-run this extractor.")

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
               grid_s=args.grid_s, dtype=args.dtype,
               compression=args.compression, overwrite=args.overwrite)

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
        print(f'[FATAL] {err} subject(s) failed; the grid embedding cache is '
              f'INCOMPLETE. Fix and re-run before training with '
              f'--egobrain_use_grid_embeddings.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
