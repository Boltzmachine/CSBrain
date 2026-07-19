#!/bin/bash

#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=motion_patch
#SBATCH --output=outputs/slurms/%j.out

# Finish the motion cache for the newly-videoed EgoBrain subjects: only the
# `patch` space remains (all 16 `pixel` tasks completed in job 2092836).
#
# WHY THIS RERUN EXISTS — job 2092836 DEADLOCKED after 19/32 tasks:
#   * every worker sat at 0.0% CPU, ~0 bytes/s (rchar), and held ZERO open .h5
#     file descriptors, for >15 min. Not slow -- parked.
#   * HDF5 file locking was ON (HDF5_USE_FILE_LOCKING unset) and 16 forked
#     workers were opening large HDF5 files on GPFS. HDF5's flock-based locking
#     on a distributed filesystem is a well-known hang; h5py is also not
#     fork-safe, so a big Pool multiplies the exposure.
# Mitigations here:
#   1. HDF5_USE_FILE_LOCKING=FALSE   (the actual fix for GPFS)
#   2. --workers 4 (was 16)          -- far less lock/IO contention; the job is
#      bandwidth-bound anyway, so concurrency bought little.
#   3. --spaces patch                -- pixel is already complete; don't redo it.
#   4. --block 1000 (was 3000)       -- smaller reads, lower peak RSS.
#
# The builder skips any subject whose output .npy already exists, so the 24
# original subjects and the 3 already-done new ones (P0025/P0027/P0039) are
# never rewritten. Production reads <sub>_patch_cos_d5.npy.
#
# NOTE the interleaved emb layout (grid chunk = (1,2,P,d)) means the patch
# reader pulls BOTH orientations though it only needs orient-0 -> ~2x read
# amplification (~602 GB of chunks for ~301 GB of useful data). One-time cost.
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
SUBS=$(python3 -c "print(','.join('P%04d'%i for i in range(25,41)))")
exec zsh -lc "
  module load miniconda
  conda activate cbramod
  export HDF5_USE_FILE_LOCKING=FALSE
  echo '[motion-patch] HDF5_USE_FILE_LOCKING='\$HDF5_USE_FILE_LOCKING
  echo '[motion-patch] subjects: $SUBS'
  exec python -m scripts.build_egobrain_motion_cache \
      --subjects '$SUBS' \
      --spaces patch \
      --step_slots 5 \
      --block 1000 \
      --workers 4
"
