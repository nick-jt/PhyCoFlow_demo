#!/bin/bash
#SBATCH --job-name=kolm_sweep_fleet
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=96G
#SBATCH --output=kolm_sweep_fleet_%j.log
# Sensor-count sweep for the NON-DMF-Gen Kolmogorov rows, so fig:sensors panel
# (a) carries the whole 2D fleet instead of one curve. Densities match the
# DMF-Gen sweep and the classical sweep exactly: 65/164/655/1965/6554
# (0.1/0.25/1/3/10% of 256^2). One JSON per (model, density).
set -uo pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
STM=../Save_TrainedModel/kolmogorov2d
FAIL=0
run () {  # run <model> <run-dir glob> [extra flags]
  local M=$1 G=$2; shift 2
  local RUN; RUN=$(readlink -f "$(ls -d $G 2>/dev/null | tail -1)" 2>/dev/null)
  if [ -z "$RUN" ] || [ ! -f "$RUN/best.pt" ]; then echo "[skip] $M: no run dir"; FAIL=1; return; fi
  echo "=== $M -> $RUN"
  python eval_kolm_ensemble.py --model "$M" --run-dir "$RUN" --ckpt best --K 8 \
      --n-obs-list 65 164 655 1965 6554 --cond-fields 0 --expect-val-len 640 \
      --stratify-blocks 1 --out-prefix kolm_sweep --resume --n-frames 50 \
      --seed 0 --op-seed 1000 --no-figs "$@" || { echo "[fail] $M"; FAIL=1; }
}
run senseiver "$STM/baseline_det/Baseline_senseiver_Stage1_DemoN61_*"
run mlp_rbf   "$STM/baseline_mlp_rbf/Baseline_mlp_rbf_Stage1_DemoN64_*"
run geofno    "$STM/baseline_geofno/Baseline_geofno_Stage1_DemoN65_*"
run sit       "$STM/baseline_sit/Baseline_sit_Stage1_DemoN62_*"
run latent_fm "$STM/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN60_*" --nfe 4
run s3gm      "$STM/baseline_s3gm/Baseline_s3gm_Stage1_DemoN63_*"
exit $FAIL
