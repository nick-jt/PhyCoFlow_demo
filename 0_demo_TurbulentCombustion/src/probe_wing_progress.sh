#!/bin/bash
#SBATCH --job-name=wing_probe
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=48G
#SBATCH --output=wing_probe_%j.log

# Mid-flight sanity probe: score an IN-PROGRESS run's last.pt on the real
# protocol metric, on a couple of cases.
#
# Why this exists: training/validation loss does not predict protocol
# rel-L2. MLP-RBF finished 4000 epochs at val 0.1913 (indistinguishable from
# Senseiver's 0.1916, which scores 0.422) and evaluated to 1.086, worse than
# the train-mean floor. Fourteen GPU-hours were spent before that was
# visible. Run this every few hours against anything long.
#
# env: MODEL (senseiver|mlp_rbf|sit|dmfgen), RUN_DIR, NCASES (default 2)
set -euo pipefail
source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

MODEL=${MODEL:?set MODEL}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
NCASES=${NCASES:-2}

echo "[probe] $MODEL last.pt, $NCASES cases, $(date)"
python -u eval_wing_ensemble.py --model "$MODEL" --run-dir "$RUN_DIR" \
    --ckpt last --n-cases "$NCASES" 2>&1 | tail -25
echo "[probe] reference: DMF-Gen 0.436 | Senseiver 0.422 | gappy POD 0.625 | floor 1.000"
