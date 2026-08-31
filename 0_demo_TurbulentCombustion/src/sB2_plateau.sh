#!/bin/bash
#SBATCH --job-name=cnfB2_plateau
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=128G
# Stage-B fix 2: continue arm-P stage-1 (5.18M decoder) to held-out codec
# plateau (patience-based stop, TUNE-split odd val snapshots only).
set -euo pipefail
JT=/home/ntricard/.claude/jobs/3ac3fd02/tmp/confild_improve
ROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
BC=$ROOT/Save_TrainedModel/JHU/baseline_confild
RESUME=$BC/unified_published_prior/Baseline_confild_Stage1_DemoN23_20260828_182524/last.pt
OUT=$BC/improve/stage1_plateau_P
source ~/envs/jhtdb
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral MPLCONFIGDIR=/tmp
echo "node: $(hostname)"

# Best-effort NVMe staging of the 5.9 GB HDF5 (same rationale as the unified
# trainer: insulate startup I/O from Lustre contention).
SRC=/projects/ammoniacomb/generative_reconstruction/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5
STAGE_DIR="/tmp/${USER}/jhu_${SLURM_JOB_ID}"
DATA=""
cleanup() { rm -rf "$STAGE_DIR" 2>/dev/null || true; }
trap cleanup EXIT
if mkdir -p "$STAGE_DIR" 2>/dev/null && cp "$SRC" "$STAGE_DIR/" 2>/dev/null; then
  if [ "$(stat -c%s "$SRC")" -eq "$(stat -c%s "$STAGE_DIR/$(basename "$SRC")")" ]; then
    DATA="$STAGE_DIR/$(basename "$SRC")"
    echo "staged to $DATA"
  fi
fi

python "$JT/confild_stage1_plateau.py" \
  --resume-ckpt "$RESUME" --out-dir "$OUT" ${DATA:+--data "$DATA"} \
  --eval-every 15 --patience 12 --min-delta 0.002 \
  --val-offsets 1 21 41 --tta-steps 400 --max-hours 22.0
echo "=== B2 plateau done ==="
