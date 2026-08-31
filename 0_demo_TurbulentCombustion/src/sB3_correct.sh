#!/bin/bash
#SBATCH --job-name=cnfB3_corr
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
# Stage-B fix 3: tune the post-DPS sensor-consistency correction on the TUNE
# split (at the tuned DPS scale from Stage A iii), then run the full-50 K=8
# eval of "CoNFiLD-P+ (existing ckpts, tuned scale, sensor-corrected)".
set -u
JT=/home/ntricard/.claude/jobs/3ac3fd02/tmp/confild_improve
ROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
BC=$ROOT/Save_TrainedModel/JHU/baseline_confild
S1=$BC/unified_published_prior/Baseline_confild_Stage1_DemoN23_20260828_182524/best.pt
S2=$BC/unified_published_prior/Baseline_confild_Stage2_DemoN23_20260829_075611/best.pt
TUNE_OUT=$BC/improve/tune_P
OUT=$BC/improve/plus_existing_P
source ~/envs/jhtdb
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 MPLCONFIGDIR=/tmp TQDM_DISABLE=1
mkdir -p "$OUT"
echo "node: $(hostname)"

SCALE=$(python -c "import json;print(json.load(open('$TUNE_OUT/Evaluation/best_scale_P.json'))['best']['dps_scale'])" 2>/dev/null || echo 1.0)
echo "tuned DPS scale: $SCALE"

for COMBO in "0 0.005" "25 0.003" "25 0.01" "100 0.003" "100 0.01"; do
  set -- $COMBO; ST=$1; LR=$2
  echo "=== correction steps=$ST lr=$LR ==="
  python "$JT/confild_eval_unified2.py" \
    --stage1-ckpt "$S1" --stage2-ckpt "$S2" --out-dir "$OUT" \
    --tag "tuneB3_s${ST}_lr${LR}" --snaps 1 7 13 19 25 31 --K 2 \
    --dps-scale "$SCALE" --post-sensor-steps "$ST" --post-sensor-lr "$LR" \
    || echo "WARN: combo $ST/$LR failed"
done
python "$JT/confild_pick_best.py" --eval-dir "$OUT/Evaluation" \
  --prefix tuneB3_ --name corr_P

BEST_ST=$(python -c "import json;print(json.load(open('$OUT/Evaluation/best_corr_P.json'))['best']['post_sensor_steps'])")
BEST_LR=$(python -c "import json;print(json.load(open('$OUT/Evaluation/best_corr_P.json'))['best']['post_sensor_lr'])")
echo "=== final full-50 K=8: scale=$SCALE corr=$BEST_ST@$BEST_LR ==="
python "$JT/confild_eval_unified2.py" \
  --stage1-ckpt "$S1" --stage2-ckpt "$S2" --out-dir "$OUT" \
  --tag PplusEx --n-snapshots 50 --K 8 --dps-scale "$SCALE" \
  --post-sensor-steps "$BEST_ST" --post-sensor-lr "$BEST_LR"
python "$JT/confild_split_summary.py" --eval-dir "$OUT/Evaluation" --tag PplusEx \
  --label "CoNFiLD-P+ (existing ckpts, tuned scale, sensor-corrected)"
echo "=== B3 done ==="
