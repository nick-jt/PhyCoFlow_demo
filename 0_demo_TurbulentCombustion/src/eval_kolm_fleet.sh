#!/bin/bash
#SBATCH --job-name=kolm_fleet
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --output=eval_kolm_fleet_%j.log

# Matched-protocol 2D Kolmogorov fleet eval. All sampling goes through
# src/eval_kolm_ensemble.py (canonical helpers.build_sparse_condition sensor
# draw under torch.manual_seed(seed*777+snap); fixed 50 evenly-spaced val
# frames; K=8 for generative models, deterministic rows tiled per the JHU
# canonical convention). See that file's docstring for why the ENSEMBLE_K
# hook is not used for the 2D fleet.
#
# GENERIC LAUNCHER: the MODELS env var selects the legs, e.g.
#   MODELS=senseiver sbatch eval_kolm_fleet.sh                 # 1h backfill
#   MODELS="latent_fm sit dmfgen" sbatch --time=03:30:00 eval_kolm_fleet.sh
#   MODELS="geofno mlprbf s3gm" sbatch eval_kolm_fleet.sh      # once trained
#                                                              # (needs driver legs)
# Per-model run dirs are resolved from the latest matching directory; override
# with RUN_DIR_<MODEL> (uppercased), e.g. RUN_DIR_SENSEIVER=/path/to/run.
# Wall default is 1h (single-leg backfill); pass sbatch --time for more.
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

MODELS=${MODELS:-"latent_fm sit dmfgen senseiver"}

resolve_run_dir() {
  # $1 = model name. RUN_DIR_<MODEL> env wins; else latest matching glob.
  local m=$1 ovr pat
  ovr=$(eval echo "\${RUN_DIR_$(echo "$m" | tr '[:lower:]' '[:upper:]')-}")
  if [ -n "$ovr" ]; then readlink -f "$ovr"; return; fi
  case $m in
    dmfgen)     pat="$STM/pointcloud_ffm/bench_kolm_v1_DemoN*" ;;
    latent_fm)  pat="$STM/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN*" ;;
    sit)        pat="$STM/baseline_sit/Baseline_sit_Stage1_DemoN*" ;;
    senseiver)  pat="$STM/baseline_det/Baseline_senseiver_Stage1_DemoN*" ;;
    geofno)     pat="$STM/baseline_geofno/Baseline_geofno_Stage1_DemoN*" ;;
    mlprbf)     pat="$STM/baseline_mlprbf/Baseline_mlp_rbf_Stage1_DemoN*" ;;
    s3gm)       pat="$STM/baseline_s3gm/Baseline_s3gm_Stage1_DemoN*" ;;
    *)          echo ""; return ;;
  esac
  readlink -f "$(ls -d $pat 2>/dev/null | tail -1)" 2>/dev/null || echo ""
}

model_flags() {
  # Extra eval_kolm_ensemble.py flags per model (beyond the shared protocol).
  case $1 in
    dmfgen)    echo "--nfe 4 --n-obs-list 65 164 655 1965 6554" ;;
    latent_fm) echo "--nfe 4 --n-obs-list 655" ;;
    *)         echo "--n-obs-list 655" ;;  # sit: config sampling_N; senseiver: det
  esac
}

echo "=== node $(hostname) job ${SLURM_JOB_ID} MODELS='$MODELS' ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

FAIL=0
for M in $MODELS; do
  RUN=$(resolve_run_dir "$M")
  if [ -z "$RUN" ] || [ ! -d "$RUN" ]; then
    echo "[launcher] $M: no run dir found (set RUN_DIR_$(echo "$M" | tr '[:lower:]' '[:upper:]') or train it first); SKIPPING as FAILURE"
    FAIL=1
    continue
  fi
  echo "=== $M -> $RUN ==="
  # shellcheck disable=SC2046
  if ! python eval_kolm_ensemble.py --model "$M" --run-dir "$RUN" \
        --ckpt best --K 8 $(model_flags "$M") \
        --n-frames 50 --seed 0 --op-seed 1000 --fig-every 10; then
    echo "[launcher] $M EVAL FAILED"
    FAIL=1
  fi
  echo "--- $M JSONs ---"
  ls -la "$RUN"/Evaluation/kolm_fleet_*.json \
         "$RUN"/Evaluation/sensor_sweep_*.json 2>/dev/null || true
  echo "$M crps files: $(ls "$RUN"/Evaluation/crps_snap*.json 2>/dev/null | wc -l)"
done

exit $FAIL
