#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
# Stage A (iii): DPS guidance-scale sweep on the TUNE split (odd snaps in
# window 0), K=2. usage: sbatch sA_sweep.sh <ARM> <stage1-best> <stage2-best> <out-dir>
set -u
JT=/home/ntricard/.claude/jobs/3ac3fd02/tmp/confild_improve
ARM="${1:?arm}"; S1="${2:?stage1 ckpt}"; S2="${3:?stage2 ckpt}"; OUT="${4:?out dir}"
source ~/envs/jhtdb
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 MPLCONFIGDIR=/tmp TQDM_DISABLE=1
mkdir -p "$OUT"
echo "node: $(hostname)  arm: $ARM"
for SC in 0.3 1.0 3.0 10.0 30.0; do
  echo "=== arm $ARM scale $SC ==="
  python "$JT/confild_eval_unified2.py" \
    --stage1-ckpt "$S1" --stage2-ckpt "$S2" --out-dir "$OUT" \
    --tag "tuneA_scale${SC}" --snaps 1 7 13 19 25 31 --K 2 \
    --dps-scale "$SC" || echo "WARN: scale $SC failed for arm $ARM"
done
python "$JT/confild_pick_best.py" --eval-dir "$OUT/Evaluation" \
  --prefix tuneA_scale --name "scale_${ARM}"
echo "=== sweep $ARM done ==="
