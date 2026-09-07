#!/bin/bash
#SBATCH --job-name=dump_kolm_gallery
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --output=dump_kolm_gallery_%j.log

# Reconstruction-gallery field dumps: ONE fixed val frame per 2D dataset,
# identical canonical sensors across models, via eval_kolm_ensemble.py's
# --dump-frame mode (added for this figure; the mode reuses the eval's own
# loaders, sensor draw torch.manual_seed(seed*777+snap) and ensemble-noise
# base seed*131+si, so the dumped sample IS eval sample k=0 of the fleet
# JSONs when the frame is canonical).
#
#   Kolmogorov : val frame 256 (canonical eval frame, si=20; absolute 2816),
#                n_obs 655 on vorticity, K=8; NFE 4 (dmfgen/latent_fm),
#                SiT its configured sampling_N=50, Senseiver deterministic.
#                Expected sensor fingerprint (from the fleet JSONs):
#                sensors=655 idx_sum=22128955.
#   Cylinder   : val frame 300 (canonical, si=25; absolute 1500, held-out
#                Re250 tail), cond_fields 0 1 (Ux,Uy; p UNOBSERVED),
#                n_obs per the run configs: 238/field on the 23,800-pt mesh
#                (dmfgen, senseiver), 800/field on the 400x200 grid
#                (sit) -- both 1% of N.
#
# PENDING (no leg possible today):
#   * kolm mlprbf / geofno / s3gm : eval_kolm_ensemble.py has no driver leg
#     for these families yet.
#   * cylinder latent_fm : only the Stage1 autoencoder checkpoint exists
#     (Baseline_latent_fm_Stage1_DemoN70); no Stage2 flow to sample.
#   * cylinder mlprbf / geofno / s3gm : same driver gap as kolm.
#
# Each leg is INDEPENDENT (a preempted job leaves finished npz usable) and
# skipped when its npz already exists, so the script is resubmittable to
# backfill panels as checkpoints land.

set -uo pipefail
export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/envs/jhtdb

WT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion
cd "$WT/src"
KOLM=$WT/Save_TrainedModel/kolmogorov2d
CYL=$WT/Save_TrainedModel/cylinder2d
OUT=$WT/Save_TrainedModel_pof/field_dumps
mkdir -p "$OUT"

echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} $(date) ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

FAIL=0
run_dump () {  # $1 tag; rest: eval_kolm_ensemble.py args
  local TAG=$1; shift
  if [ -s "$OUT/$TAG.npz" ]; then
    echo "##### $TAG already present, skipping"; return 0
  fi
  echo "##### $TAG start $(date +%T)"
  if python eval_kolm_ensemble.py "$@" --dump-npz "$OUT/$TAG.npz"; then
    echo "##### $TAG OK $(date +%T)"
  else
    echo "##### $TAG FAILED rc=$? $(date +%T)"; FAIL=1
  fi
}

# ---------------- Kolmogorov, val frame 256 (n_obs 655, seed 0) --------------
run_dump kolm_senseiver --model senseiver \
  --run-dir "$KOLM/baseline_det/Baseline_senseiver_Stage1_DemoN61_20260905_161349" \
  --ckpt best --dump-frame 256 --n-obs-list 655 --seed 0

run_dump kolm_latent_fm --model latent_fm \
  --run-dir "$KOLM/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN60_20260905_143200" \
  --ckpt best --dump-frame 256 --n-obs-list 655 --K 8 --nfe 4 --seed 0

run_dump kolm_dmfgen --model dmfgen \
  --run-dir "$KOLM/pointcloud_ffm/bench_kolm_v1_DemoN101_20260905_122940" \
  --ckpt best --dump-frame 256 --n-obs-list 655 --K 8 --nfe 4 --seed 0

run_dump kolm_sit --model sit \
  --run-dir "$KOLM/baseline_sit/Baseline_sit_Stage1_DemoN62_20260905_122941" \
  --ckpt best --dump-frame 256 --n-obs-list 655 --K 8 --seed 0
  # no --nfe: the run config's sampling_N (=50), as in the fleet eval

# ---------------- Cylinder, val frame 300 (cond Ux,Uy; p unobserved) ---------
# Fleet started 2026-09-06 ~08:20; best.pt exists for all legs below (dumps
# read a best-so-far checkpoint if training is still running -- rerun this
# script after the fleet finishes to refresh: delete the npz first).
run_dump cyl_senseiver --model senseiver \
  --run-dir "$CYL/baseline_det/Baseline_senseiver_Stage1_DemoN71_20260906_082017" \
  --ckpt best --dump-frame 300 --n-obs-list 238 --cond-fields 0 1 \
  --expect-val-len 600 --seed 0

run_dump cyl_dmfgen --model dmfgen \
  --run-dir "$CYL/pointcloud_ffm/bench_cyl_v1_DemoN102_20260906_082022" \
  --ckpt best --dump-frame 300 --n-obs-list 238 --cond-fields 0 1 \
  --expect-val-len 600 --K 8 --nfe 4 --seed 0

run_dump cyl_sit --model sit \
  --run-dir "$CYL/baseline_sit/Baseline_sit_Stage1_DemoN73_20260906_082303" \
  --ckpt best --dump-frame 300 --n-obs-list 800 --cond-fields 0 1 \
  --expect-val-len 600 --K 8 --seed 0

echo "=== done $(date); files:"
ls -la "$OUT"
exit $FAIL
