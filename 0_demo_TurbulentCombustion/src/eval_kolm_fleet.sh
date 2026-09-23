#!/bin/bash
#SBATCH --job-name=fleet_eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
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

WT=/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd "$WT/src"

DATASET=${DATASET:-kolmogorov2d}
STM=$WT/Save_TrainedModel/$DATASET
# Seed replicates (kolmogorov2d_seed7, cylinder2d_seed1337, ...) share the
# base dataset's protocol constants; only the run root and prefix differ.
BASE=${DATASET%%_seed*}
SEEDTAG=${DATASET#"$BASE"}

case $BASE in
  kolmogorov2d)
    EXPECT=640; NOBS=655; COND="0"; BLOCKS=1; PREFIX=kolm_fleet$SEEDTAG; CONDSRC=points
    MODELS=${MODELS:-"latent_fm sit dmfgen senseiver mlp_rbf geofno s3gm"} ;;
  cylinder2d)
    # 600-frame val block = held-out Re {80, 250}, 300 frames each ->
    # frames stratified across the two Re sub-blocks. cond_fields [0,1]
    # (Ux, Uy observed; p unobserved = identifiability probe).
    EXPECT=600; NOBS=238; COND="0 1"; BLOCKS=2; PREFIX=cyl_fleet$SEEDTAG; CONDSRC=points
    MODELS=${MODELS:-"dmfgen senseiver mlp_rbf geofno sit s3gm"} ;;
  kolmogorov2d_fullbudget)
    # Full-budget reruns of the two short Kolmogorov rows (mlp_rbf, s3gm).
    EXPECT=640; NOBS=655; COND="0"; BLOCKS=1; PREFIX=kolm_fleet_full; CONDSRC=points
    MODELS=${MODELS:-"mlp_rbf s3gm"} ;;
  cylinder2d_uonly)
    # Observe-u-only cross-channel variant (P2-10): Ux observed at 1%; Uy AND p
    # unobserved. Runs live under Save_TrainedModel/cylinder2d_uonly.
    EXPECT=600; NOBS=238; COND="0"; BLOCKS=2; PREFIX=cyl_uonly; CONDSRC=points
    MODELS=${MODELS:-"dmfgen senseiver"} ;;
  cylinder2d_surface)
    # Surface-to-field task: sensors ONLY on the wall-adjacent ring
    # (mesh: 360 cells; grid: their unique nearest fluid cells), all three
    # fields observed at the taps, tap budget swept 32/64/128/360 per field
    # (capped at the pool). Runs live under Save_TrainedModel/cylinder2d_surface.
    EXPECT=600; NOBS="32 64 128 360"; COND="0 1 2"; BLOCKS=2; PREFIX=cyl_surface; CONDSRC=surface
    MODELS=${MODELS:-"dmfgen senseiver mlp_rbf geofno sit s3gm latent_fm"} ;;
  *) echo "[launcher] unknown DATASET=$DATASET"; exit 1 ;;
esac

# NOBS_OVERRIDE lets a caller evaluate a subset of the density sweep. Used for
# the surface task's grid-locked models: their sensor pool is 62 cells, so any
# request above 62 draws the identical pool and 128/360 reproduce 64 bit for
# bit (verified for geofno/sit/latent_fm). Re-running them is pure cost.
# OPERATOR passes measurement-operator flags straight to the evaluator, e.g.
#   OPERATOR="--sensor-noise 0.1"      OPERATOR="--sensor-occlusion 0.25"
# The evaluator puts the operator into the cache key, the output filename and
# the protocol stamp itself, so nothing here needs to rename anything -- which
# is the point of having fixed it there rather than in each launcher.
OPERATOR="${OPERATOR:-}"
if [ -n "$OPERATOR" ]; then
  echo "[launcher] OPERATOR=$OPERATOR (evaluator stamps and names the outputs)"
fi

if [ -n "${NOBS_OVERRIDE:-}" ]; then
  # A density override must NOT write to the canonical output name: the
  # canonical JSON is the fleet's headline row at the fleet's density, and an
  # override run at a different density silently replaced it once. Redirect the
  # prefix so override runs land beside the canonical row, never on top of it.
  # The density goes in the NAME, not just a marker. A bare "_ovr" prefix is
  # still one filename for every density, so consecutive override runs
  # overwrite each other -- which is the same defect one level down from the
  # canonical row it was added to protect. Encode the actual sensor count.
  NOBS=$NOBS_OVERRIDE
  PREFIX="${PREFIX}_ovr_n$(echo "$NOBS_OVERRIDE" | tr ' ' 'x')"
  echo "[launcher] NOBS_OVERRIDE=$NOBS_OVERRIDE -> writing with prefix '$PREFIX' (canonical row untouched)"
fi

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
      if [ "$BASE" = kolmogorov2d ] && [ -z "$SEEDTAG" ]; then
        echo "--nfe 4 --n-obs-list 65 164 655 1965 6554"   # sensor sweep (canonical seed only)
      else
        echo "--nfe 4 --n-obs-list $NOBS"
      fi ;;
    latent_fm) echo "--nfe 4 --n-obs-list $NOBS" ;;
    *)         echo "--n-obs-list $NOBS" ;;  # sit/s3gm: config sampling_N; (surface: NOBS is the tap sweep)
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
        --cond-fields $COND --expect-val-len $EXPECT --cond-source $CONDSRC \
        --stratify-blocks $BLOCKS --out-prefix $PREFIX --resume \
        --n-frames 50 --seed 0 --op-seed 1000 --fig-every 10 $OPERATOR; then
    echo "[launcher] $M EVAL FAILED"
    FAIL=1
  fi
  echo "--- $M JSONs ---"
  ls -la "$RUN"/Evaluation/${PREFIX}*.json "$RUN"/Evaluation/${PREFIX}*_n*.json \
         "$RUN"/Evaluation/sensor_sweep_*.json 2>/dev/null || true
  echo "$M crps files: $(ls "$RUN"/Evaluation/crps_snap*.json 2>/dev/null | wc -l)"
done

exit $FAIL
