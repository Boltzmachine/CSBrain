#!/bin/bash
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --job-name=split-probe
#SBATCH --output=outputs/slurms/%j.out

# Interpretability sweep for the frame-averaging BILATERALIZATION SPLIT.
#
# Runs scripts/probe_bilateral_split.py over the wm-new* family in BOTH domains:
#   egobrain -- the PRETRAINING domain. Any claim about what the gate LEARNED has
#               to hold here; PhysioNet is transfer.
#   physio   -- the downstream MI task (adds the mi/swap stages, which need labels).
#
# Sweep: the gradneg epoch trajectory (is the split strengthening or eroding?), a
# random-init control (the gate at init has weight=0, bias=-2 -> a CONSTANT gate,
# which is exactly the analytic null for the antisym probe), and the three sibling
# runs (does the gradient-negative contrastive term compete with the split?).
#
# NOTE: conda must be active BEFORE sbatch (this script does not self-activate).

set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=outputs/split_probe
mkdir -p "$OUT"

GRADNEG1=outputs/wm-new-gradneg/epoch1_loss30.862201690673828.pth
GRADNEG2=outputs/wm-new-gradneg/epoch2_loss2.571162700653076.pth
GRADNEG3=outputs/wm-new-gradneg/epoch3_loss2.50658917427063.pth
GRADNEG5=outputs/wm-new-gradneg/epoch5_loss2.414652109146118.pth
GRADNEG10=outputs/wm-new-gradneg/epoch10_loss2.3018484115600586.pth
WMNEW10=outputs/wm-new/epoch10_loss2.2622766494750977.pth
BIG10=outputs/wm-new-biggradneg/epoch10_loss2.378716230392456.pth
ALPHA10=outputs/wm-new-alpha1/epoch10_loss2.4680819511413574.pth

# ---- in-domain (EgoBrain): weights / antisym / eqconst ----
ego() {
    echo "=================== EGO $2 ==================="
    python -u scripts/probe_bilateral_split.py --ckpt "$1" --tag "ego_$2" \
        --domain egobrain --stages weights,antisym,eqconst \
        --max_batches 40 --ego_clips 4000 --num_workers 0 \
        --out "$OUT/ego_$2.json" "${@:3}" || echo "FAILED: ego_$2"
}
ego "$GRADNEG1"  gradneg_ep1
ego "$GRADNEG2"  gradneg_ep2
ego "$GRADNEG3"  gradneg_ep3
ego "$GRADNEG5"  gradneg_ep5
ego "$GRADNEG10" gradneg_ep10
ego "$GRADNEG10" randinit --random_init
ego "$WMNEW10"   wmnew_ep10
ego "$BIG10"     biggradneg_ep10
ego "$ALPHA10"   alpha1_ep10

# ---- transfer (PhysioNet-MI): + the labelled mi / swap stages ----
phy() {
    echo "=================== PHY $2 ==================="
    python -u scripts/probe_bilateral_split.py --ckpt "$1" --tag "phy_$2" \
        --domain physio --stages weights,antisym,eqconst,mi,swap \
        --max_batches 24 --num_workers 2 \
        --out "$OUT/phy_$2.json" "${@:3}" || echo "FAILED: phy_$2"
}
phy "$GRADNEG1"  gradneg_ep1
phy "$GRADNEG5"  gradneg_ep5
phy "$GRADNEG10" gradneg_ep10
phy "$GRADNEG10" randinit --random_init
phy "$WMNEW10"   wmnew_ep10
phy "$BIG10"     biggradneg_ep10
phy "$ALPHA10"   alpha1_ep10

echo "ALL DONE"
ls -la "$OUT"
