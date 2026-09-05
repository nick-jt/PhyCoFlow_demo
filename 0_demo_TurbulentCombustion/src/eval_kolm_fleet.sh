#!/bin/bash
#SBATCH --job-name=kolm_fleet
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --output=eval_kolm_fleet_%j.log

# Matched-protocol 2D Kolmogorov fleet eval (DMF-Gen + latent-FM + SiT) plus
# the DMF-Gen sensor-count sweep. All sampling goes through
# src/eval_kolm_ensemble.py (canonical helpers.build_sparse_condition sensor
# draw under torch.manual_seed(seed*777+snap); fixed 50 evenly-spaced val
# frames; K=8). See that file's docstring for why the ENSEMBLE_K hook is not
# used for the 2D fleet (helpers_baseline RNG divergence, unseeded draws, and
# the hook never firing for the SiT patch tokenizer).
#
# NOTE: deliberately no trailing `echo "exit status: $?"` -- that pattern
# forces exit 0 and makes a crashed job report COMPLETED.

set -uo pipefail

# Split env MUST match training (trajectory-holdout val block, no gap).
export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/envs/jhtdb

WT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion
cd "$WT/src"

STM=$WT/Save_TrainedModel/kolmogorov2d
DMF=$(readlink -f "$(ls -d $STM/pointcloud_ffm/bench_kolm_v1_DemoN101_* | tail -1)")
LFM=$(readlink -f "$(ls -d $STM/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN60_* | tail -1)")
SIT=$(readlink -f "$(ls -d $STM/baseline_sit/Baseline_sit_Stage1_DemoN62_* | tail -1)")

echo "=== node $(hostname) job ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "DMF=$DMF"
echo "LFM=$LFM"
echo "SIT=$SIT"

FAIL=0

# --- baselines first (fast): 50 frames, n_obs=655, K=8 ----------------------
# latent-FM: NFE=4 (matched flow-model setting). SiT: no --nfe, so it uses
# its configured benchmark step count (run config sampling_N = 50).
echo "=== latent-FM ==="
if ! python eval_kolm_ensemble.py --model latent_fm --run-dir "$LFM" \
      --ckpt best --K 8 --nfe 4 --n-obs-list 655 \
      --n-frames 50 --seed 0 --op-seed 1000 --fig-every 10; then
  echo "[launcher] LATENT-FM EVAL FAILED"; FAIL=1
fi

echo "=== SiT ==="
if ! python eval_kolm_ensemble.py --model sit --run-dir "$SIT" \
      --ckpt best --K 8 --n-obs-list 655 \
      --n-frames 50 --seed 0 --op-seed 1000 --fig-every 10; then
  echo "[launcher] SIT EVAL FAILED"; FAIL=1
fi

# --- DMF-Gen: main protocol (n=655) + full sensor sweep in one process ------
echo "=== DMF-Gen sweep ==="
if ! python eval_kolm_ensemble.py --model dmfgen --run-dir "$DMF" \
      --ckpt best --K 8 --nfe 4 --n-obs-list 65 164 655 1965 6554 \
      --n-frames 50 --seed 0 --op-seed 1000 --fig-every 10; then
  echo "[launcher] DMF-GEN EVAL FAILED"; FAIL=1
fi

echo "=== JSONs written ==="
ls -la "$DMF"/Evaluation/sensor_sweep_dmfgen*.json \
       "$DMF"/Evaluation/kolm_fleet_dmfgen_*.json 2>/dev/null || true
ls "$LFM"/Evaluation/kolm_fleet_latentfm_*.json \
   "$SIT"/Evaluation/kolm_fleet_sit_*.json 2>/dev/null || true
echo "LFM crps files: $(ls "$LFM"/Evaluation/crps_snap*.json 2>/dev/null | wc -l)"
echo "SiT crps files: $(ls "$SIT"/Evaluation/crps_snap*.json 2>/dev/null | wc -l)"
exit $FAIL
