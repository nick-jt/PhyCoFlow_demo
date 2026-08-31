#!/bin/bash
#SBATCH --job-name=cnfB1b_sel
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
# Stage-B fix 1: TUNE-split checkpoint selection + full-50 K=8 final eval.
# usage: sbatch sB1b_select.sh <stage1-ckpt> <s2dir> <out-dir> <final-tag> <label> [scale_json] [corr_json]
set -euo pipefail
JT=/home/ntricard/.claude/jobs/3ac3fd02/tmp/confild_improve
S1="${1:?}"; S2DIR="${2:?}"; OUT="${3:?}"; TAG="${4:?}"; LABEL="${5:?}"
SCALE_JSON="${6:-}"; CORR_JSON="${7:-}"
source ~/envs/jhtdb
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 MPLCONFIGDIR=/tmp TQDM_DISABLE=1
echo "node: $(hostname)"
python "$JT/confild_select_and_final.py" \
  --stage1-ckpt "$S1" --s2dir "$S2DIR" --out-dir "$OUT" \
  --tune-snaps 1 7 13 19 25 31 --k-tune 2 \
  ${SCALE_JSON:+--scale-from "$SCALE_JSON"} \
  ${CORR_JSON:+--corr-from "$CORR_JSON"} \
  --final-K 8 --final-n 50 --final-tag "$TAG" --label "$LABEL"
echo "=== select+final done ==="
