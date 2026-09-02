#!/bin/bash
#SBATCH --job-name=env_check
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --output=env_check_%j.out
# Environment guard for the Engaging campaign. Run this BEFORE launching
# training — a broken env otherwise surfaces as a job that dies 30 s in with a
# misleading traceback (see ~/envs/phycoflow for the 2026-09-01 incident).
#
#   bash check_env.sh     -> LOGIN-SAFE checks only (metadata; no imports)
#   sbatch check_env.sh   -> the above PLUS real imports on a compute node
#
# Why the split: Engaging LOGIN nodes interrupt any process that loads numpy's
# or torch's compiled core, so `import numpy` cannot be tested there at all
# (this affects ~/envs/warp-env too — it is a site restriction, not our env).
# `pip check` needs no imports, and would have caught the original scipy/numpy
# ABI mismatch, so it is the guard that actually runs where we launch from.
set -u
source ~/envs/phycoflow

FAIL=0
echo "=== python ==="
python -V
echo "interpreter: $(which python)"

echo
echo "=== dependency consistency (pip check) ==="
# This is the check that catches version-skew bugs like scipy-vs-numpy.
if pip check; then
  echo "PASS: no broken requirements"
else
  echo "FAIL: broken requirements above — fix before launching"
  FAIL=1
fi

echo
echo "=== required packages present (metadata only, no import) ==="
python - <<'PY' || FAIL=1
import importlib.metadata as md
import sys
required = ["torch","numpy","scipy","h5py","matplotlib","pyyaml","tqdm",
            "pykeops","neuraloperator",
            # needed by the UPSTREAM CoNFiLD checkout, not by this repo's src/.
            # Missed on the first pass because the inventory only scanned src/,
            # and the canonical DPS eval died on `import einops` as a result.
            "einops","pillow","blobfile"]
missing = []
for p in required:
    try:
        print(f"  {p:16s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p:16s} MISSING")
        missing.append(p)
if missing:
    print("FAIL: missing packages:", ", ".join(missing))
    sys.exit(1)
print("PASS: all required packages present")
PY

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo
  echo "=== import checks SKIPPED (login node) ==="
  echo "Login nodes cannot load numpy/torch compiled cores. To verify imports:"
  echo "    sbatch $(basename "$0")"
  exit $FAIL
fi

echo
echo "=== real imports (compute node $(hostname)) ==="
DEMO=/home/ntricard/projects/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $DEMO/src
export CONFILD_ROOT=${CONFILD_ROOT:-/orcd/scratch/orcd/002/ntricard/baselines/CoNFiLD}
module load cuda/12.4.0
python - <<'PY' || FAIL=1
import sys, traceback
# The exact import that broke the campaign, plus both trainer entry paths.
targets = [
    ("scipy.ndimage",        "from scipy.ndimage import binary_dilation, distance_transform_edt"),
    ("helpers_baseline",     "import helpers_baseline"),
    ("helpers",              "import helpers"),
    ("model_baseline",       "import model_baseline"),
    ("train_Gen_Baseline",   "import train_Gen_Baseline"),
    ("train_Det_Baseline",   "import train_Det_Baseline"),
    ("Model",                "import Model"),
    ("train_pointcloud_ffm", "import train_pointcloud_ffm"),
    ("ensemble_eval",        "import ensemble_eval"),
    # Eval entry points, not just trainers: confild_eval_unified pulls the
    # UPSTREAM checkout onto sys.path at import time and needs its deps
    # (einops, ...), which no trainer touches. Importing only trainers let a
    # broken canonical eval reach the queue.
    ("confild_eval_unified", "import confild_eval_unified"),
    ("evaluate_confild_stage1", "import evaluate_confild_stage1"),
]
bad = []
for name, stmt in targets:
    try:
        exec(stmt)
        print(f"  OK    {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        bad.append(name)
if bad:
    print("FAIL: could not import:", ", ".join(bad)); sys.exit(1)
print("PASS: all campaign imports succeed")
PY

echo
echo "=== torch/CUDA ==="
python -c "
import torch
print('  torch', torch.__version__, '| cuda available:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')
" || FAIL=1

echo
echo "=== KeOps runtime JIT (Model.py kNN search depends on this) ==="
# KeOps compiles kernels at RUNTIME, not import, and logs a benign-looking
# 'cannot find -lnvrtc' link warning. Import success therefore proves nothing —
# run a real GPU reduction so a JIT failure surfaces here, not 20 minutes into
# a 13.5 h training job.
python - <<'PY' || FAIL=1
import sys, torch
try:
    from pykeops.torch import LazyTensor
    x = torch.randn(500, 3, device="cuda")
    y = torch.randn(400, 3, device="cuda")
    D = ((LazyTensor(x[:, None, :]) - LazyTensor(y[None, :, :])) ** 2).sum(-1)
    idx = D.argKmin(8, dim=1)                      # the kNN op Model.py uses
    ref = torch.cdist(x, y).topk(8, largest=False).indices
    ok = torch.equal(idx.sort(dim=1).values, ref.sort(dim=1).values)
    print(f"  KeOps argKmin -> {tuple(idx.shape)}; matches torch.cdist: {ok}")
    if not ok:
        print("FAIL: KeOps kNN disagrees with reference"); sys.exit(1)
    print("PASS: KeOps JIT works on GPU")
except Exception as e:
    import traceback; traceback.print_exc(limit=4)
    print(f"FAIL: KeOps runtime broken: {type(e).__name__}: {e}"); sys.exit(1)
PY

echo
[ "$FAIL" -eq 0 ] && echo "ENV CHECK PASSED" || echo "ENV CHECK FAILED"
exit $FAIL
