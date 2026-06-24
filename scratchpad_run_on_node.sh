#!/bin/bash
# Helper to run a command in the wilor env on the debug node via ssh.
# Usage: ssh r4516u05n01 "bash -l /abs/path/scratchpad_run_on_node.sh <cmd...>"
module load miniconda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wilor
cd /gpfs/radev/pi/ying_rex/wq44/CSBrain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$@"
