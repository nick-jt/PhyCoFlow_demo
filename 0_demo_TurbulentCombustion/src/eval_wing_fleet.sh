#!/bin/bash
#SBATCH --job-name=eval_wing_fleet
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --output=eval_wing_fleet_%j.log
#
# NOT YET RUN. Submit only after the three learned baselines have finished
# training (jobs wing_senseiver / wing_mlprbf / wing_sit) -- every leg needs a
# best.pt and this script FAILS a leg whose run dir or checkpoint is missing,
# so a premature submission reports the gap instead of silently skipping it.
#
# Matched-protocol SHIFT-WING fleet eval. All sampling goes through
# src/eval_wing_ensemble.py, which imports evaluate_wing.build_surface_obs
# VERBATIM: 512 taps + 3x128 wall-shear sensors + 2 exact Mach/alpha tokens,
# per-case sensor seed seed*100+case, on the first 8 validation cases -- i.e.
# exactly the 8 inverse problems and 8 sensor layouts our model was scored on.
# Metrics are ensemble_eval.ensemble_metrics, reported per field
# (Ux, Uy, Uz, Cp) and as the aggregate. Deterministic rows are tiled into two
# identical members (CRPS == MAE exactly, dispersion fields null, K = 1).
#
#   MODELS="senseiver mlp_rbf sit dmfgen" sbatch eval_wing_fleet.sh
#   RUN_DIR_SIT=/path/to/run sbatch eval_wing_fleet.sh    # override a glob
#   DRY=1 bash eval_wing_fleet.sh                         # CPU plan check
#
# Reference to reproduce on the dmfgen leg (K=4, NFE=4, these 8 cases):
#   aggregate rel-L2 0.436, CRPS 0.161,
#   per-field Ux 0.645 / Uy 0.482 / Uz 0.519 / Cp 0.0995.
#
# NOTE: deliberately no trailing `echo "exit status: $?"` -- that pattern
# forces exit 0 and makes a crashed job report COMPLETED.

set -uo pipefail

# Must match training: our v3 run did NOT export WING_AUGMENT, and in any case
# ShiftWingDataset only augments the train split, never val.
unset WING_AUGMENT || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/envs/jhtdb

WT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion
cd "$WT/src"

STM=$WT/Save_TrainedModel/wing
# Our model's run lives in the main checkout, not the worktree.
MAIN=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion

MODELS=${MODELS:-"senseiver mlp_rbf sit dmfgen"}
NCASES=${NCASES:-8}
SEED=${SEED:-0}
NTAPS=${NTAPS:-512}
NSHEAR=${NSHEAR:-128}
KGEN=${KGEN:-4}
DRY=${DRY:-0}

resolve_run_dir() {
  local m=$1 ovr pat
  ovr=$(eval echo "\${RUN_DIR_$(echo "$m" | tr '[:lower:]' '[:upper:]')-}")
  if [ -n "$ovr" ]; then readlink -f "$ovr"; return; fi
  case $m in
    dmfgen)    pat="$MAIN/Save_TrainedModel/wing/pointcloud_ffm/iclr_wing_v3_expanded_DemoN13_*" ;;
    senseiver) pat="$STM/baseline_senseiver/Baseline_senseiver_Stage1_DemoN50_*" ;;
    mlp_rbf)   pat="$STM/baseline_mlp_rbf/Baseline_mlp_rbf_Stage1_DemoN51_*" ;;
    sit)       pat="$STM/baseline_sit/Baseline_sit_Stage1_DemoN52_*" ;;
    *)         echo ""; return ;;
  esac
  readlink -f "$(ls -d $pat 2>/dev/null | tail -1)" 2>/dev/null || echo ""
}

model_flags() {
  # dmfgen reproduces the reference row at K=4 / NFE=4. sit takes its own
  # configured sampling_N unless NFE is exported. The deterministic rows
  # ignore K and nfe entirely (one forward, det-tiled).
  case $1 in
    dmfgen) echo "--K $KGEN --nfe ${NFE:-4}" ;;
    sit)    echo "--K $KGEN ${NFE:+--nfe $NFE}" ;;
    *)      echo "" ;;
  esac
}

echo "=== node $(hostname) job ${SLURM_JOB_ID:-dry} MODELS='$MODELS' ==="
if [ "$DRY" = "0" ]; then
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
fi

FAIL=0
for M in $MODELS; do
  RUN=$(resolve_run_dir "$M")
  UP=$(echo "$M" | tr '[:lower:]' '[:upper:]')
  if [ -z "$RUN" ] || [ ! -d "$RUN" ]; then
    echo "[launcher] $M: no run dir found (set RUN_DIR_$UP or train it first); FAILURE"
    FAIL=1
    continue
  fi
  CK=best.pt
  if [ ! -f "$RUN/$CK" ]; then
    echo "[launcher] $M: $RUN has no $CK (training incomplete?); FAILURE"
    FAIL=1
    continue
  fi
  echo "=== $M -> $RUN ==="
  EXTRA=""
  [ "$DRY" = "1" ] && EXTRA="--dry-run"
  # shellcheck disable=SC2046,SC2086
  if ! python eval_wing_ensemble.py --model "$M" --run-dir "$RUN" \
        --ckpt best --split val --n-cases "$NCASES" --seed "$SEED" \
        --n-taps "$NTAPS" --n-shear "$NSHEAR" \
        --expect-val-len 73 --out-prefix wing_fleet \
        $(model_flags "$M") $EXTRA; then
    echo "[launcher] $M EVAL FAILED"
    FAIL=1
  fi
  echo "--- $M JSONs ---"
  ls -la "$RUN"/Evaluation/wing_fleet_*.json 2>/dev/null || true
done

exit $FAIL
