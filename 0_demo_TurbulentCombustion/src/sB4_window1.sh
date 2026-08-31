#!/bin/bash
#SBATCH --job-name=cnfB4_win1
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
# Stage-B fix 4: window=1 INFORMATION arm on the existing arm-P checkpoints --
# each snapshot reconstructed from a joint window sample in which only the
# target row carries sensors (temporal information matched to the single-frame
# baselines). Canonical scale 1.0 to isolate the window effect. Full 50, K=8.
set -euo pipefail
JT=/home/ntricard/.claude/jobs/3ac3fd02/tmp/confild_improve
ROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
BC=$ROOT/Save_TrainedModel/JHU/baseline_confild
S1=$BC/unified_published_prior/Baseline_confild_Stage1_DemoN23_20260828_182524/best.pt
S2=$BC/unified_published_prior/Baseline_confild_Stage2_DemoN23_20260829_075611/best.pt
OUT=$BC/improve/window1_P
source ~/envs/jhtdb
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 MPLCONFIGDIR=/tmp TQDM_DISABLE=1
mkdir -p "$OUT"
echo "node: $(hostname)"
python "$JT/confild_eval_unified2.py" \
  --stage1-ckpt "$S1" --stage2-ckpt "$S2" --out-dir "$OUT" \
  --tag window1 --n-snapshots 50 --K 8 --dps-scale 1.0 --window1
python "$JT/confild_split_summary.py" --eval-dir "$OUT/Evaluation" --tag window1 \
  --label "CoNFiLD-P (window=1 information, existing ckpts)"
echo "=== B4 done ==="
