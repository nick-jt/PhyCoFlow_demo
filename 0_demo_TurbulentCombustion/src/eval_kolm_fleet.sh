#!/bin/bash
#SBATCH --job-name=fleet_eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --output=eval_fleet_%j.log

# Matched-protocol 2D fleet eval (Kolmogorov / cylinder). All sampling goes
# through src/eval_kolm_ensemble.py (canonical helpers.build_sparse_condition
# sensor draw under torch.manual_seed(seed*777+snap); fixed evenly-spaced val
# frames; K=8 generative, det-tiled deterministic rows; see its docstring).
#
# GENERIC LAUNCHER:
#   DATASET  kolmogorov2d (default) | cylinder2d
#   MODELS   space-separated legs, e.g.
#     MODELS="mlp_rbf s3gm" sbatch eval_kolm_fleet.sh
#     DATASET=cylinder2d MODELS="senseiver mlp_rbf geofno sit dmfgen s3gm" \
#         sbatch eval_kolm_fleet.sh
#     DATASET=cylinder2d MODELS=latent_fm sbatch eval_kolm_fleet.sh   # once
#         its stage 2 lands (leg pre-registered below)
#   RUN_DIR_<MODEL> overrides the per-model run-dir glob.
# Wall default is 2h (backfill); pass sbatch --time for more. Every leg runs
# with --resume, so a timed-out job can simply be resubmitted and continues
# from the crps_snapN.json files already written.
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

DATASET=${DATASET:-kolmogorov2d}
STM=$WT/Save_TrainedModel/$DATASET

case $DATASET in
  kolmogorov2d)
    EXPECT=640; NOBS=655; COND="0"; BLOCKS=1; PREFIX=kolm_fleet
    MODELS=${MODELS:-"latent_fm sit dmfgen senseiver mlp_rbf geofno s3gm"} ;;
  cylinder2d)
    # 600-frame val block = held-out Re {80, 250}, 300 frames each ->
    # frames stratified across the two Re sub-blocks. cond_fields [0,1]
    # (Ux, Uy observed; p unobserved = identifiability probe).
    EXPECT=600; NOBS=238; COND="0 1"; BLOCKS=2; PREFIX=cyl_fleet
    MODELS=${MODELS:-"dmfgen senseiver mlp_rbf geofno sit s3gm"} ;;
  *) echo "[launcher] unknown DATASET=$DATASET"; exit 1 ;;
esac

resolve_run_dir() {
  # $1 = model name. RUN_DIR_<MODEL> env wins; else latest matching glob.
  local m=$1 ovr pat
  ovr=$(eval echo "\${RUN_DIR_$(echo "$m" | tr '[:lower:]' '[:upper:]')-}")
  if [ -n "$ovr" ]; then readlink -f "$ovr"; return; fi
  case $m in
    dmfgen)     pat="$STM/pointcloud_ffm/bench_*_DemoN*" ;;
    latent_fm)  pat="$STM/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN*" ;;
    sit)        pat="$STM/baseline_sit/Baseline_sit_Stage1_DemoN*" ;;
    senseiver)  pat="$STM/baseline_det/Baseline_senseiver_Stage1_DemoN*" ;;
    mlp_rbf|mlprbf) pat="$STM/baseline_mlp_rbf/Baseline_mlp_rbf_Stage1_DemoN*" ;;
    geofno)     pat="$STM/baseline_geofno/Baseline_geofno_Stage1_DemoN*" ;;
    s3gm)       pat="$STM/baseline_s3gm/Baseline_s3gm_Stage1_DemoN*" ;;
    *)          echo ""; return ;;
  esac
  readlink -f "$(ls -d $pat 2>/dev/null | tail -1)" 2>/dev/null || echo ""
}

model_flags() {
  # Extra eval_kolm_ensemble.py flags per model (beyond the shared protocol).
  case $1 in
    dmfgen)
      if [ "$DATASET" = kolmogorov2d ]; then
        echo "--nfe 4 --n-obs-list 65 164 655 1965 6554"   # sensor sweep
      else
        echo "--nfe 4 --n-obs-list $NOBS"
      fi ;;
    latent_fm) echo "--nfe 4 --n-obs-list $NOBS" ;;
    *)         echo "--n-obs-list $NOBS" ;;  # sit/s3gm: config sampling_N;
                                             # senseiver/mlp_rbf/geofno: det
  esac
}

echo "=== node $(hostname) job ${SLURM_JOB_ID} DATASET=$DATASET MODELS='$MODELS' ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

FAIL=0
for M in $MODELS; do
  RUN=$(resolve_run_dir "$M")
  if [ -z "$RUN" ] || [ ! -d "$RUN" ]; then
    echo "[launcher] $M: no run dir found (set RUN_DIR_$(echo "$M" | tr '[:lower:]' '[:upper:]') or train it first); SKIPPING as FAILURE"
    FAIL=1
    continue
  fi
  if [ ! -f "$RUN/best.pt" ]; then
    echo "[launcher] $M: $RUN has no best.pt (training incomplete?); SKIPPING as FAILURE"
    FAIL=1
    continue
  fi
  echo "=== $M -> $RUN ==="
  # shellcheck disable=SC2046,SC2086
  if ! python eval_kolm_ensemble.py --model "$M" --run-dir "$RUN" \
        --ckpt best --K 8 $(model_flags "$M") \
        --cond-fields $COND --expect-val-len $EXPECT \
        --stratify-blocks $BLOCKS --out-prefix $PREFIX --resume \
        --n-frames 50 --seed 0 --op-seed 1000 --fig-every 10; then
    echo "[launcher] $M EVAL FAILED"
    FAIL=1
  fi
  echo "--- $M JSONs ---"
  ls -la "$RUN"/Evaluation/${PREFIX}_*.json \
         "$RUN"/Evaluation/sensor_sweep_*.json 2>/dev/null || true
  echo "$M crps files: $(ls "$RUN"/Evaluation/crps_snap*.json 2>/dev/null | wc -l)"
done

exit $FAIL
