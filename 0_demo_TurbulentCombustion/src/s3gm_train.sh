#!/bin/bash
#SBATCH --job-name=s3gm_train
#SBATCH --time=21:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
# S3GM baseline, JHU cross-cube. Capacity-matched: 6,348,228 trainable params
# (-2.43% of the 6,506,253-param comparison model).
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
source ~/envs/jhtdb
BASE=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
cd "$BASE/src"
CFGROOT="$BASE/Save_config"
# Parameterised so a smoke run validates this exact staging code path.
CFG_IN="${S3GM_CONFIG:-$CFGROOT/config_baseline_S3GM_xcube.yaml}"
LOG="s3gm_train_${SLURM_JOB_ID}.log"

# ---------------------------------------------------------------------------
# Stage the H5 to node-local NVMe.
#
# TurbulentCombustionH5Dataset.__getitem__ reads the 5.9 GB H5 on EVERY access
# with no caching, so a 150-snapshot epoch moves ~4.7 GB. On a contended shared
# Lustre that turns the wall-clock budget into a function of who else is
# running, which breaks budget-matching as a controlled variable.
#
# Any failure falls back to the shared path -- staging is an optimisation, never
# a correctness dependency.
# ---------------------------------------------------------------------------
SHARED_H5=$(python - "$CFG_IN" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["shared"]["paths"]["data_path"])
PY
)
STAGE_DIR="/tmp/$USER/jhu_${SLURM_JOB_ID}"
DATA_PATH="$SHARED_H5"
cleanup() { rm -rf "$STAGE_DIR" 2>/dev/null || true; }
trap cleanup EXIT

{
  echo "[stage] shared source: $SHARED_H5"
  SRC_SIZE=$(stat -c%s "$SHARED_H5" 2>/dev/null || echo 0)
  if ! mkdir -p "$STAGE_DIR" 2>/dev/null; then
    echo "[stage] cannot create $STAGE_DIR; FALLBACK to shared path"
  elif [ "$SRC_SIZE" -le 0 ]; then
    echo "[stage] source unreadable or empty; FALLBACK to shared path"
  else
    AVAIL=$(df -B1 --output=avail "$STAGE_DIR" 2>/dev/null | tail -1)
    AVAIL=${AVAIL:-0}
    NEED=$(( SRC_SIZE + 10737418240 ))          # source + 10 GB headroom
    if [ "$AVAIL" -lt "$NEED" ]; then
      echo "[stage] only ${AVAIL}B free, need ${NEED}B; FALLBACK to shared path"
    else
      T0=$SECONDS
      if cp "$SHARED_H5" "$STAGE_DIR/data.h5" 2>/dev/null; then
        DST_SIZE=$(stat -c%s "$STAGE_DIR/data.h5" 2>/dev/null || echo 0)
        if [ "$DST_SIZE" -eq "$SRC_SIZE" ]; then
          DATA_PATH="$STAGE_DIR/data.h5"
          echo "[stage] OK: ${SRC_SIZE} bytes -> $DATA_PATH in $((SECONDS-T0))s"
        else
          echo "[stage] SIZE MISMATCH ($DST_SIZE != $SRC_SIZE); FALLBACK to shared path"
          rm -f "$STAGE_DIR/data.h5"
        fi
      else
        echo "[stage] copy FAILED; FALLBACK to shared path"
      fi
    fi
  fi
  echo "[stage] data_path in use: $DATA_PATH"
} >> "$LOG" 2>&1

# Job-local config: data_path -> staged copy, canonical path preserved as
# data_path_shared so eval_s3gm3d.py can fall back on another node.
CFG_JOB="$STAGE_DIR/config_s3gm_job.yaml"
if ! python - "$CFG_IN" "$CFG_JOB" "$DATA_PATH" "$SHARED_H5" >> "$LOG" 2>&1 <<'PY'
import sys, yaml
cfg_in, cfg_out, data_path, shared = sys.argv[1:5]
c = yaml.safe_load(open(cfg_in))
c["shared"]["paths"]["data_path"] = data_path
c["shared"]["paths"]["data_path_shared"] = shared
yaml.safe_dump(c, open(cfg_out, "w"), sort_keys=False)
print(f"[stage] wrote job config {cfg_out}")
PY
then
  echo "[stage] job-config write FAILED; using the checked-in config unchanged" >> "$LOG"
  CFG_JOB="$CFG_IN"
fi

# NOTE: deliberately no trailing `echo "exit status: $?"` -- that pattern makes
# the batch script exit 0 so a crashed run reports COMPLETED to sacct.
CUDA_VISIBLE_DEVICES=0 python train_s3gm3d.py \
    --config "$CFG_JOB" --training-stage 1 >> "$LOG" 2>&1
RC=$?
echo "train rc=$RC" >> "$LOG"

# Run-level duty cycle: summed GPU compute against SLURM elapsed.
python - "$LOG" "$SLURM_JOB_ID" >> "$LOG" 2>&1 <<'PY'
import re, subprocess, sys, json
log, job = sys.argv[1], sys.argv[2]
txt = open(log, errors="replace").read().replace("\r", "\n")
comp = [float(x) for x in re.findall(r"compute_s=([0-9.]+)", txt)]
data = [float(x) for x in re.findall(r"data_s=([0-9.]+)", txt)]
steps = [int(x) for x in re.findall(r"opt_steps_total=(\d+)", txt)]
el = subprocess.run(["sacct","-j",job,"--format=ElapsedRaw","-n","-P"],
                    capture_output=True, text=True).stdout.split("\n")[0].strip()
el = float(el) if el.isdigit() else 0.0
tc, td = sum(comp), sum(data)
print(f"[duty] opt_steps={steps[-1] if steps else 0} compute_s={tc:.1f} "
      f"loader_wait_s={td:.1f} slurm_elapsed_s={el:.1f} "
      f"epoch_duty={tc/max(tc+td,1e-9):.3f} job_duty={tc/max(el,1e-9):.3f}")
PY

if [ "$RC" -ne 0 ]; then exit "$RC"; fi
