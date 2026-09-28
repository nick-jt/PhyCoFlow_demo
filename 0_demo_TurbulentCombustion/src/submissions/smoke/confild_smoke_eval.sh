#!/bin/bash
#SBATCH --job-name=cnf_eval_smoke
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=96G
set -euo pipefail
REPO=/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion
source ~/envs/jhtdb
cd "$REPO/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 MPLCONFIGDIR=/tmp
S1=$(cat /home/ntricard/.claude/jobs/3ac3fd02/tmp/armC_rundir.txt)/best.pt
echo "stage1: $S1"
python confild_eval_unified.py \
  --stage1-ckpt "$S1" \
  --out-dir /home/ntricard/.claude/jobs/3ac3fd02/tmp/eval_smoke \
  --steps 20 --K 2 --n-snapshots 4 --row-chunk 4
