#!/bin/bash
#SBATCH --job-name=fno3d_matched
#SBATCH --time=21:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=128G

set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

DEMO_NUM=90
CFG=Save_config/config_baseline_fno3d_xcube.yaml
BUDGET_S=70200                      # 19.5 h of training wall clock
ARCHIVE_START_S=66600               # 18h30m -> 10 archives x 6 min over the last 5%
LOG_FILE="train_fno3d_matched_${SLURM_JOB_ID}_DemoN${DEMO_NUM}.log"
JOB_T0=$(date +%s)
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, python: $(which python)" > "$LOG_FILE"

SD=$(grep -oP '^save_dir\s*:\s*"\K[^"]+' ../${CFG})
if ls -d ../${SD}_DemoN${DEMO_NUM}_* >/dev/null 2>&1; then
  echo "ABORT: ../${SD}_DemoN${DEMO_NUM}_* already exists" >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Stage the 5.9 GB HDF5 to node-local NVMe. This backbone is the most I/O-heavy
# in the fleet: requires_full_grid means every __getitem__ reads a whole
# 1,953,125-point snapshot (~31 MB) with no dataset-level caching, so on a
# contended Lustre mount the epoch is dominated by read wait. Falls back to the
# shared path if anything about the copy is wrong.
# ---------------------------------------------------------------------------
SHARED_DATA=$(grep -oP '^data\s*:\s*"\K[^"]+' ../${CFG})
DATA_PATH=$(bash stage_h5.sh "$SHARED_DATA" "$SLURM_JOB_ID" 2>>"$LOG_FILE")
echo "[stage] training will read: $DATA_PATH" >> "$LOG_FILE"

# Job-local config with the staged path. args.json is rewritten back to the
# shared path below so downstream evaluation can still find the data.
JOB_CFG_ABS="/tmp/${USER}/jhu_${SLURM_JOB_ID}_config.yaml"
python - "$SHARED_DATA" "$DATA_PATH" "../${CFG}" "$JOB_CFG_ABS" <<'PY'
import sys
shared, local, src, dst = sys.argv[1:5]
s = open(src).read()
if local != shared:
    assert s.count(f'"{shared}"') == 1
    s = s.replace(f'"{shared}"', f'"{local}"')
open(dst, "w").write(s)
print(f"[stage] job config -> {dst}")
PY

# ---------------------------------------------------------------------------
# Background caretaker: repair args.json's data path as soon as it appears, and
# archive ~10 checkpoints over the last 5% of the budget.
# ---------------------------------------------------------------------------
(
  for _ in $(seq 1 240); do
    RD=$(ls -d ../${SD}_DemoN${DEMO_NUM}_* 2>/dev/null | head -1)
    if [ -n "${RD:-}" ] && [ -f "$RD/args.json" ]; then
      python - "$RD/args.json" "$DATA_PATH" "$SHARED_DATA" <<'PY'
import json, sys
p, local, shared = sys.argv[1:4]
c = json.load(open(p))
if c.get("data") == local and local != shared:
    c["data"] = shared
    json.dump(c, open(p, "w"), indent=2)
PY
      break
    fi
    sleep 15
  done
  sleep $(( ARCHIVE_START_S > 3600 ? ARCHIVE_START_S - 3600 : 1 ))
  while [ $(( $(date +%s) - JOB_T0 )) -lt $ARCHIVE_START_S ]; do sleep 60; done
  for i in $(seq 1 10); do
    RD=$(ls -d ../${SD}_DemoN${DEMO_NUM}_* 2>/dev/null | head -1)
    if [ -n "${RD:-}" ] && [ -f "$RD/last.pt" ]; then
      mkdir -p "$RD/archive"
      cp "$RD/last.pt" "$RD/archive/last_t$(printf '%02d' $i).pt" 2>/dev/null
      echo "[archive] $i at $(( $(date +%s) - JOB_T0 )) s" >> "$LOG_FILE"
    fi
    sleep 360
  done
) &
CARE_PID=$!

# Wall-clock-bounded budget: the epoch count is deliberately larger than
# reachable, and `timeout` stops training at exactly 19.5 h. Exit 124 is a
# planned budget stop, not a crash.
CUDA_VISIBLE_DEVICES=0 timeout --signal=INT ${BUDGET_S} \
  python -u train_pointcloud_ffm.py \
  --config "$JOB_CFG_ABS" --Demo-Num ${DEMO_NUM} >> "$LOG_FILE" 2>&1
RC=$?
kill $CARE_PID 2>/dev/null
ELAPSED=$(( $(date +%s) - JOB_T0 ))
if [ $RC -eq 124 ] || [ $RC -eq 130 ]; then
  echo "[launcher] planned budget stop at ${BUDGET_S}s (rc=$RC)" >> "$LOG_FILE"
else
  echo "[launcher] trainer return code: $RC" >> "$LOG_FILE"
fi

# ---------------------------------------------------------------------------
# Cost summary: optimizer steps alongside wall clock, and the duty cycle
# (summed compute vs SLURM elapsed). PURE_STEP_S is the data-resident
# per-optimizer-step compute time from bench_fno3d.py; if it has not been
# measured yet the duty cycle is left null and filled in by the eval job.
# ---------------------------------------------------------------------------
RD=$(ls -d ../${SD}_DemoN${DEMO_NUM}_* 2>/dev/null | head -1)
if [ -n "${RD:-}" ]; then
  python - "$RD" "$ELAPSED" "$DATA_PATH" "$SHARED_DATA" "$SLURM_JOB_ID" >> "$LOG_FILE" 2>&1 <<'PY'
import csv, glob, json, os, sys
rd, elapsed, local, shared, jobid = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
rows = []
for c in glob.glob(os.path.join(rd, "Loss_*", "losses.csv")):
    rows = list(csv.DictReader(open(c)))
n_train = 150
bs = 8
spe = -(-n_train // bs)
epochs = len(rows)
cum = float(rows[-1]["cumul_train_time_s"]) if rows else 0.0
ep_t = [float(r["epoch_time_s"]) for r in rows]
med = sorted(ep_t)[len(ep_t)//2] if ep_t else None
out = {
    "job_id": jobid,
    "data_path_used": local,
    "staged_to_local_nvme": local != shared,
    "slurm_elapsed_s": elapsed,
    "epochs_completed": epochs,
    "steps_per_epoch": spe,
    "optimizer_steps": epochs * spe,
    "median_epoch_time_s": med,
    "train_loop_seconds_total": cum,
    "train_loop_fraction_of_elapsed": (cum / elapsed) if elapsed else None,
    "pure_compute_step_seconds": None,
    "duty_cycle_compute_over_elapsed": None,
    "note": ("pure_compute_step_seconds and duty_cycle are filled in by "
             "bench_fno3d.py (data-resident measurement) in the eval job"),
}
ci = os.path.join(rd, "cost_instrumentation.json")
if os.path.exists(ci):
    b = json.load(open(ci))
    s = b.get("train_step_seconds")
    if s:
        out["pure_compute_step_seconds"] = s
        out["duty_cycle_compute_over_elapsed"] = (s * out["optimizer_steps"]) / elapsed
json.dump(out, open(os.path.join(rd, "run_cost_summary.json"), "w"), indent=1)
print("[runcost] " + " ".join(f"{k}={v}" for k, v in out.items() if k != "note"))
PY
fi

rm -rf "/tmp/${USER}/jhu_${SLURM_JOB_ID}" "$JOB_CFG_ABS" 2>/dev/null
exit $RC
