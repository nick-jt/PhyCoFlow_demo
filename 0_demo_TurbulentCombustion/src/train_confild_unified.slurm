#!/bin/bash
#SBATCH --job-name=confild_unified
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=128G
#
# usage: sbatch train_confild_unified.slurm <config-basename>
#
# Runs stage 1 then stage 2 in ONE allocation so the ~19.5 h wall-clock budget
# is a single matched budget across both stages. Per-stage cut-offs are in the
# YAML (stage1 13.5 h + stage2 5.5-6.0 h).
#
# NOTE: no trailing `echo "exit status: $?"` anywhere. That idiom forces exit 0
# and makes a crashed run report COMPLETED.

set -euo pipefail

REPO=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
CONFIG="${1:?usage: sbatch train_confild_unified.slurm <config-basename>}"

source ~/envs/jhtdb
cd "$REPO"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export JHU_AUGMENT=octahedral
export MPLCONFIGDIR=/tmp
# One stamp for the whole allocation; run_stage re-stamps CONFILD_SLURM_START
# per stage so each stage's duty cycle uses its own denominator.
export CONFILD_JOB_START=$(date +%s)

echo "=== CoNFiLD unified run ==="
echo "job        : ${SLURM_JOB_ID}"
echo "config     : ${CONFIG}"
echo "node       : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# Stage the 5.9 GB HDF5 to node-local NVMe.
#
# The shared Lustre path is read by every job in the fleet at once; a sibling
# baseline measured a 45% duty cycle and a 13-minute stall against it. Our
# stage-1 hot loop is insulated (cache_snapshots preloads every snapshot into
# RAM once, measured 99.9% duty cycle inside the loop), but STARTUP still
# streams ~9.4 GB and all three arms start together, so they would contend with
# each other. Staging removes that coupling.
#
# Strictly best-effort: any failure -- no /tmp space, short copy, sed failure --
# falls back to the shared path, so this can only help.
# ---------------------------------------------------------------------------
CONFIG_PATH="Save_config/${CONFIG}"
RUN_CONFIG="$CONFIG_PATH"
SRC=$(awk '/^[[:space:]]*data_path:/ {print $2; exit}' "$CONFIG_PATH")
STAGE_DIR="/tmp/${USER}/jhu_${SLURM_JOB_ID}"
cleanup() { rm -rf "$STAGE_DIR" 2>/dev/null || true; }
trap cleanup EXIT

stage_ok=0
if [ -f "$SRC" ] && mkdir -p "$STAGE_DIR" 2>/dev/null; then
  echo "staging    : $SRC -> $STAGE_DIR"
  t0=$(date +%s)
  if cp "$SRC" "$STAGE_DIR/" 2>/dev/null; then
    LOCAL="$STAGE_DIR/$(basename "$SRC")"
    src_sz=$(stat -c%s "$SRC" 2>/dev/null || echo 0)
    dst_sz=$(stat -c%s "$LOCAL" 2>/dev/null || echo 1)
    if [ "$src_sz" -gt 0 ] && [ "$src_sz" -eq "$dst_sz" ]; then
      CAND="$STAGE_DIR/$(basename "$CONFIG_PATH")"
      if sed "s|^\([[:space:]]*data_path:[[:space:]]*\).*|\1${LOCAL}|" "$CONFIG_PATH" > "$CAND" \
         && grep -q "data_path:[[:space:]]*${LOCAL}" "$CAND"; then
        RUN_CONFIG="$CAND"; stage_ok=1
        echo "staging    : OK ($((src_sz/1024/1024)) MB in $(( $(date +%s) - t0 )) s)"
      fi
    fi
  fi
fi
[ "$stage_ok" -eq 1 ] || echo "staging    : FALLBACK to shared path (staging unavailable or incomplete)"
echo "run config : $RUN_CONFIG"
echo "data_path  : $(awk '/^[[:space:]]*data_path:/ {print $2; exit}' "$RUN_CONFIG")"

run_stage () {
  local stage=$1
  echo "=== stage ${stage} ==="
  local t0=$(date +%s)
  export CONFILD_SLURM_START=$t0
  python src/train_Gen_Baseline.py --config "$RUN_CONFIG" --training-stage "$stage" --device cuda:0
  echo "[confild:slurm] stage=${stage} slurm_stage_elapsed_s=$(( $(date +%s) - t0 )) staged=${stage_ok}"
}

run_stage 1
run_stage 2

echo "[confild:slurm] job_total_elapsed_s=$(( $(date +%s) - CONFILD_JOB_START )) staged=${stage_ok}"
echo "=== done ==="
