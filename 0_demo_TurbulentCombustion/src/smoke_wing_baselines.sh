#!/bin/bash
#SBATCH --job-name=smoke_wing_base
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --output=smoke_wing_base_%j.log
#
# 3-epoch smoke gate for the three SHIFT-WING learned baselines
# (senseiver, mlp_rbf, sit/pointnet), all conditioned on the BODY-SURFACE
# observation pool.
#
# THIS SCRIPT IS THE FLEET GATE. The full trainings are submitted with
# --dependency=afterok on it, so it MUST be able to fail. Hence:
#   * `set -euo pipefail` at the top -- the campaign has already been bitten
#     by a smoke script that piped python through `tail`, which swallowed the
#     exit status and made a crashed job report COMPLETED;
#   * NO trailing `echo "exit status: $?"` (that also forces exit 0);
#   * every python invocation writes straight to the job log, unpiped.
#
# What it verifies, per model:
#   1. training completes 3 epochs without an exception;
#   2. the `[train] epoch=... loss=... time=... peak_mem=...` cost
#      instrumentation is present for all 3 epochs;
#   3. all 3 losses are finite and epoch 3 < epoch 1 (descending);
#   4. the `[cond]` fingerprint reports source=surface_pool with exactly 2
#      parameter tokens and a sensor count inside the configured pool budget.
# Plus one shared, model-independent check (5) that the CANONICAL EVAL draw
# -- evaluate_wing.build_surface_obs, used verbatim by eval_wing_ensemble.py
# -- yields 512 taps + 3x128 shear + 2 parameter tokens = 898.

set -euo pipefail

# Deliberately NOT set: train_iclr_wing_v3.sh (the run these baselines are
# matched against) does not export WING_AUGMENT, so our model trained without
# spanwise reflection and the baselines must too.
unset WING_AUGMENT || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/envs/jhtdb

WT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion
cd "$WT/src"

SMOKE_ROOT="$WT/Save_TrainedModel/wing/_smoke"
SMOKE_CFG="$WT/Save_config/wing/_smoke"
mkdir -p "$SMOKE_ROOT" "$SMOKE_CFG"

echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "WING_AUGMENT=${WING_AUGMENT:-<unset, as intended>}"

# ---------------------------------------------------------------------------
# (5) canonical eval-draw check -- CPU, no model, runs first so a protocol
#     break fails the gate before an hour of GPU time is spent.
# ---------------------------------------------------------------------------
echo "=== check: evaluate_wing.build_surface_obs canonical draw ==="
python - <<'PYEOF'
import torch
from collections import Counter
from dataset_shiftwing import ShiftWingDataset, SURF_VALUE_FIELD_IDS, PARAM_FIELD_IDS
from evaluate_wing import build_surface_obs

ROOT = "/projects/ammoniacomb/generative_reconstruction/shift_wing/processed_v3"
ds = ShiftWingDataset(ROOT, split="val")
assert len(ds) == 73, f"val split is {len(ds)}, expected 73"
item = ds[0]
obs = build_surface_obs(item, n_taps=512, n_shear=128, device="cpu", seed=0)
n = obs["coords"].shape[1]
ids = Counter(obs["field_ids"][0].tolist())
print(f"[evalcheck] sensors={n} field_id_counts={dict(sorted(ids.items()))}")
assert n == 898, f"expected 898 observation tokens (512 + 3*128 + 2), got {n}"
assert ids[SURF_VALUE_FIELD_IDS[0]] == 512, ids
for fid in SURF_VALUE_FIELD_IDS[1:]:
    assert ids[fid] == 128, ids
for fid in PARAM_FIELD_IDS:
    assert ids[fid] == 1, ids
assert float(obs["mask"].sum()) == 898.0
# The parameter tokens must be LAST and exact (unperturbed).
assert obs["field_ids"][0, -2:].tolist() == list(PARAM_FIELD_IDS)
assert torch.allclose(obs["values"][0, -2:], item["param_values"])
print("[evalcheck] OK: 896 surface sensors + 2 exact parameter tokens")
PYEOF

# ---------------------------------------------------------------------------
# 3-epoch training smokes
# ---------------------------------------------------------------------------
make_smoke_cfg() {
  # $1 = source config, $2 = destination smoke config
  SRC="$1" DST="$2" SMOKE_ROOT="$SMOKE_ROOT" python - <<'PYEOF'
import os
import yaml

src, dst, smoke_root = os.environ["SRC"], os.environ["DST"], os.environ["SMOKE_ROOT"]
cfg = yaml.safe_load(open(src))
model = cfg["baseline_model"]
params = cfg[f"{model}_params"]["training"]
params["epochs"] = 3
params["eval_every"] = 1          # exercise the val pass + checkpoint write
params["save_every"] = 1000000    # keep the grid visualize path out of it
if model == "sit":
    # SiT's LR schedule is a 200-epoch linear warmup, so over 3 epochs the
    # full-run LR is ~0 and the loss trace would be pure noise -- a descent
    # gate on it would test nothing. Shorten the warmup FOR THE SMOKE ONLY so
    # the descent check is real. The full run keeps warmup_epochs=200.
    params["warmup_epochs"] = 1
cfg["shared"]["logging"]["eval_every"] = 1
cfg["shared"]["reload"] = False   # never resume a real run into the smoke dir
cfg["shared"]["paths"]["save_root"] = f"{smoke_root}/{model}"
cfg["shared"]["paths"]["config_backup_root"] = f"{smoke_root}/_cfg/{model}"
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"[smoke-cfg] {dst}  ({model}: 3 epochs, save_root={smoke_root}/{model})")
PYEOF
}

check_log() {
  # $1 = model name, $2 = log file
  MODEL="$1" LOG="$2" python - <<'PYEOF'
import math
import os
import re
import sys

model, log = os.environ["MODEL"], os.environ["LOG"]
text = open(log, encoding="utf-8", errors="replace").read()

# (2) cost instrumentation present for every epoch
pat = re.compile(r"\[train\] epoch=(\d+) loss=([0-9eE.+-]+)\s+time=([0-9.]+)s\s+"
                 r"peak_mem=(\d+)MB")
rows = pat.findall(text)
if len(rows) < 3:
    sys.exit(f"[GATE-FAIL] {model}: found {len(rows)} '[train] epoch=... "
             f"time=... peak_mem=...' lines, expected 3")
losses = [float(r[1]) for r in rows[:3]]
print(f"[gate] {model}: losses={losses} "
      f"times={[float(r[2]) for r in rows[:3]]}s "
      f"peak_mem={[int(r[3]) for r in rows[:3]]}MB")

# (3) finite and descending
if not all(math.isfinite(v) for v in losses):
    sys.exit(f"[GATE-FAIL] {model}: non-finite training loss {losses}")
if not losses[2] < losses[0]:
    sys.exit(f"[GATE-FAIL] {model}: loss did not descend "
             f"(epoch1={losses[0]:.6e} epoch3={losses[2]:.6e})")

# (4) surface-pool conditioning fingerprint
m = re.search(r"\[cond\] (\S+) source=(\S+) sensors=(\d+) "
              r"\(surface=(\d+), param_tokens=(\d+)\) per_field_id=(\{[^}]*\})",
              text)
if m is None:
    sys.exit(f"[GATE-FAIL] {model}: no '[cond] ... source=...' fingerprint "
             "in the log -- the surface-pool branch never fired")
who, source, n_tot, n_surf, n_par, per_field = m.groups()
print(f"[gate] {model}: cond source={source} sensors={n_tot} "
      f"surface={n_surf} param_tokens={n_par} per_field_id={per_field}")
if source != "surface_pool":
    sys.exit(f"[GATE-FAIL] {model}: conditioning source is {source!r}, not "
             "'surface_pool' -- sensors are being drawn from VOLUME points")
if int(n_par) != 2:
    sys.exit(f"[GATE-FAIL] {model}: {n_par} parameter tokens, expected 2")
# Training draws are random per pool channel: 64..4096 taps + 3 x 16..1024.
lo, hi = 64 + 3 * 16, 4096 + 3 * 1024
if not (lo <= int(n_surf) <= hi):
    sys.exit(f"[GATE-FAIL] {model}: {n_surf} surface sensors outside the "
             f"configured pool budget [{lo}, {hi}]")
print(f"[gate] {model}: PASS")
PYEOF
}

run_smoke() {
  # $1 = model, $2 = config basename, $3 = driver
  local MODEL="$1" CFG="$2" DRIVER="$3"
  local SCFG="$SMOKE_CFG/smoke_${MODEL}.yaml"
  local LOG="$WT/src/smoke_wing_${MODEL}_${SLURM_JOB_ID:-local}.log"
  echo "=== smoke: $MODEL ($DRIVER) ==="
  make_smoke_cfg "$WT/Save_config/$CFG" "$SCFG"
  python "$DRIVER" --config "$SCFG" --training-stage 1 --device cuda:0 2>&1 | tee "$LOG"
  check_log "$MODEL" "$LOG"
}

# `tee` above would mask python's exit status through the pipe, but
# `set -o pipefail` is in force, so a crash still fails the job. Verified by
# the check_log calls, which also fail the job on a silently-wrong run.

run_smoke senseiver config_baseline_wing_senseiver.yaml train_Det_Baseline.py
run_smoke mlp_rbf   config_baseline_wing_mlprbf.yaml    train_Det_Baseline.py
run_smoke sit       config_baseline_SiT_wing.yaml       train_Gen_Baseline.py

echo "=== ALL SMOKE GATES PASSED ==="
