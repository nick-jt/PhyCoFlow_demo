#!/bin/bash
#SBATCH --job-name=fno3d_eval
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=128G

# Canonical protocol, identical seeding to every other method:
#   --seed 0 --op-seed 1000 --n-snapshots 50 --K 8 --cond-fields 0 2 --n-obs 19531 19531
# --chunk 1953125 forces ensemble_eval's query chunking to one full-grid chunk,
# which the FNO requires. No edit to the shared ensemble_eval.py is needed.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

RD=${RD:?set RD to the run directory}
L="eval_fno3d_${SLURM_JOB_ID}.log"
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST; run dir $RD" > "$L"

# Stage to node-local NVMe so the reported wall-clock numbers are not polluted
# by Lustre contention, and repoint args.json at the local copy for this job.
SHARED=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['data'])" "$RD/args.json")
LOCAL=$(bash stage_h5.sh "$SHARED" "$SLURM_JOB_ID" 2>>"$L")
cp "$RD/args.json" "$RD/args.json.shared.bak"
python - "$RD/args.json" "$LOCAL" <<'PY'
import json, sys
p, local = sys.argv[1:3]
c = json.load(open(p)); c["data"] = local
json.dump(c, open(p, "w"), indent=2)
PY
restore() { cp "$RD/args.json.shared.bak" "$RD/args.json"; rm -rf "/tmp/${USER}/jhu_${SLURM_JOB_ID}"; }
trap restore EXIT

for CKPT in best.pt last.pt; do
  [ -f "$RD/$CKPT" ] || continue
  for NFE in 4 16; do
    TAG="${CKPT%.pt}_nfe${NFE}_K8_all50"
    python -u ensemble_eval.py \
      --run-dir "$RD" --ckpt "$CKPT" \
      --K 8 --n-steps $NFE --n-snapshots 50 \
      --n-obs 19531 19531 --cond-fields 0 2 \
      --seed 0 --op-seed 1000 --chunk 1953125 \
      --out "$RD/${TAG}.json" >> "$L" 2>&1
  done
done

# Data-resident cost instrumentation (no dataset access inside the timed loops).
python -u bench_fno3d.py --run-dir "$RD" --ckpt best.pt --nfe 4 \
  --batch-size 8 --out "$RD/cost_instrumentation.json" >> "$L" 2>&1

# Fill in the duty cycle now that pure-compute step time is known.
python - "$RD" >> "$L" 2>&1 <<'PY'
import json, os, sys
rd = sys.argv[1]
sp = os.path.join(rd, "run_cost_summary.json")
ci = os.path.join(rd, "cost_instrumentation.json")
if os.path.exists(sp) and os.path.exists(ci):
    s = json.load(open(sp)); b = json.load(open(ci))
    step = b.get("train_step_seconds")
    if step and s.get("optimizer_steps") and s.get("slurm_elapsed_s"):
        s["pure_compute_step_seconds"] = step
        s["duty_cycle_compute_over_elapsed"] = (
            step * s["optimizer_steps"] / s["slurm_elapsed_s"])
        json.dump(s, open(sp, "w"), indent=1)
    print("[runcost] " + " ".join(f"{k}={v}" for k, v in s.items() if k != "note"))
PY
