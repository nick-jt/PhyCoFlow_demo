#!/bin/bash
#SBATCH --job-name=lfm_canon
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --output=/home/ntricard/.claude/jobs/3ac3fd02/tmp/lfm_canon_%j.log

# Canonical-protocol evaluation of the FIXED latent flow-matching baseline
# (job 16997534, run Baseline_latent_fm_Stage2_DemoN24_20260828_164541).
# Submitted with --dependency=afterany:16997534 so it runs on the finished
# window checkpoints, per the WAIT rule: final numbers come from the window,
# never from mid-training checkpoints.
#
# Evaluates best.pt, last.pt, and the mid-window archive epoch_10750.pt if the
# run reached its LFM_CKPT_WINDOW (10500:11000:50). Each JSON lands in the
# run's Evaluation/ dir with 'canonical' in the filename, which
# assemble_baseline_table.py already globs
# (baseline_latent_fm/*/Evaluation/*canonical*.json).
#
# NOTE: deliberately no trailing `echo "exit status: $?"` -- that pattern
# forces exit 0 and makes a crashed job report COMPLETED.

set -uo pipefail

# Split env MUST match training (job 16997534) or the val split differs.
export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/envs/jhtdb

TMP=/home/ntricard/.claude/jobs/3ac3fd02/tmp
ROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
RUN=$ROOT/Save_TrainedModel/JHU/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN24_20260828_164541

# The eval driver's canonical home is src/eval_latentfm_ensemble.py; a scratch
# execution copy is kept next to lfm_fixes.py so this job is independent of
# checkout churn. Imports resolve against the live checkout via LFM_SRC_DIR.
EVAL=$TMP/eval_latentfm_ensemble.py
export LFM_SRC_DIR=$ROOT/src
export LFM_FIXES_DIR=$TMP

echo "=== node $(hostname) job ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

CKPTS=("best" "last")
MID=$RUN/ckpt_window/epoch_10750.pt
if [ -f "$MID" ]; then
  CKPTS+=("epoch_10750.pt")
else
  echo "[launcher] mid-window checkpoint not present ($MID); window may not have been reached"
  ls "$RUN/ckpt_window" 2>/dev/null || echo "[launcher] no ckpt_window dir at all"
fi

FAIL=0
for CK in "${CKPTS[@]}"; do
  echo "=== evaluating ckpt=$CK ==="
  if ! python "$EVAL" --run-dir "$RUN" --ckpt "$CK" \
        --K 8 --nfe 4 --n-snapshots 50 \
        --seed 0 --op-seed 1000 --cond-fields 0 2 --n-obs 19531 19531; then
    echo "[launcher] EVAL FAILED for ckpt=$CK"
    FAIL=1
  fi
done

echo "=== canonical JSONs now in $RUN/Evaluation ==="
ls -la "$RUN"/Evaluation/lfm_canonical_*.json 2>/dev/null || echo "(none)"
exit $FAIL
