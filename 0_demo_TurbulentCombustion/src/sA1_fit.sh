#!/bin/bash
#SBATCH --job-name=cnfA1_fit
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
# Stage A (i)+(ii): codec ceiling + sensor-only latent fits, arms P/C/F.
set -euo pipefail
JT=/home/ntricard/.claude/jobs/3ac3fd02/tmp/confild_improve
ROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
BC=$ROOT/Save_TrainedModel/JHU/baseline_confild
source ~/envs/jhtdb
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 MPLCONFIGDIR=/tmp TQDM_DISABLE=1
mkdir -p "$BC/improve/stageA"
echo "node: $(hostname)"
python "$JT/confild_stageA_fit.py" --arms P C F \
  --snaps 0 12 24 36 48 --steps 3000 --record-at 400 1000 2000 3000 \
  --out "$BC/improve/stageA/stageA_fit.json"
echo "=== A1 done ==="
