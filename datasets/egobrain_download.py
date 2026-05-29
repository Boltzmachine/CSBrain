"""Download the EgoBrain dataset from Hugging Face.

EgoBrain (https://huggingface.co/datasets/ut-vision/EgoBrain) is a **gated**
repository: you must request access on the dataset page and be approved
before any download will work. Once approved, set ``HF_TOKEN`` (or run
``huggingface-cli login``) and invoke this script.

The full release is ~1 TB (EEG 13.9 GB + IMU 25.4 GB + video 20-58 GB per
subject). Default download pulls **EEG + JSON metadata only**, which is
enough for the world-model recon + prediction objectives. Pass
``--with_video`` to additionally fetch the egocentric MP4s required for
the image-alignment branch.

Examples:
    # EEG only, first 3 subjects (for a debug run).
    conda run -n cbramod python -m datasets.egobrain_download \
        --subjects P0001,P0002,P0003 \
        --local_dir data/EgoBrain

    # Full release including video.
    conda run -n cbramod python -m datasets.egobrain_download \
        --subjects all --with_video --local_dir data/EgoBrain
"""

from __future__ import annotations

import argparse
import os
import sys


def _build_allow_patterns(subjects, with_video: bool, with_imu: bool):
    """Translate subject ids + modality flags into HF download globs."""
    patterns = ['README.md', 'assets/*']
    for sub in subjects:
        # Always include EEG (EDF + sidecar CSV / JSON).
        patterns += [
            f'{sub}/EmoTiv_FLEX_2_Gel/Phase_2/*.edf',
            f'{sub}/EmoTiv_FLEX_2_Gel/Phase_2/*.md.edf',
            f'{sub}/EmoTiv_FLEX_2_Gel/Phase_2/*.json',
            f'{sub}/EmoTiv_FLEX_2_Gel/Phase_2/*_intervalMarker.csv',
            f'{sub}/EmoTiv_FLEX_2_Gel/Phase_2/*_survey.csv',
        ]
        if with_video:
            patterns.append(f'{sub}/GoPro_HERO12_Ego/*.MP4')
        if with_imu:
            patterns.append(f'{sub}/WitMotion_WT9011DCL_BT50/*')
    return patterns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo_id', default='ut-vision/EgoBrain')
    p.add_argument('--local_dir', default='data/EgoBrain',
                   help='destination directory (will be created)')
    p.add_argument('--subjects', default='P0001',
                   help='comma-separated subject ids, or "all" for P0001..P0040')
    p.add_argument('--with_video', action='store_true',
                   help='also download GoPro MP4s (large: 20-58 GB / subject)')
    p.add_argument('--with_imu', action='store_true',
                   help='also download WitMotion IMU files')
    p.add_argument('--max_workers', type=int, default=8)
    args = p.parse_args()

    if args.subjects.lower() == 'all':
        subjects = [f'P{i:04d}' for i in range(1, 41)]
    else:
        subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]

    os.makedirs(args.local_dir, exist_ok=True)

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    allow_patterns = _build_allow_patterns(
        subjects, with_video=args.with_video, with_imu=args.with_imu)
    print(f'pulling {len(allow_patterns)} pattern(s) into {args.local_dir}',
          file=sys.stderr)

    try:
        out = snapshot_download(
            repo_id=args.repo_id,
            repo_type='dataset',
            local_dir=args.local_dir,
            allow_patterns=allow_patterns,
            max_workers=args.max_workers,
        )
    except GatedRepoError as e:
        sys.exit(
            f'\n[egobrain] download blocked: {e}\n'
            'Request access at https://huggingface.co/datasets/ut-vision/EgoBrain\n'
            'and ensure HF_TOKEN is set (or run `huggingface-cli login`).\n')
    print(f'done → {out}')


if __name__ == '__main__':
    main()
