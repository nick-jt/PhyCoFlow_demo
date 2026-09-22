#!/bin/bash
#SBATCH --job-name=dump_kolm_extra
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=96G
#SBATCH --output=dump_kolm_extra_%j.log
# Backfills the two Kolmogorov field dumps dump_kolm_gallery.sh listed as
# PENDING ("eval_kolm_ensemble.py has no driver leg for these families yet").
# That note is now stale: the dump path grew geofno and mlp_rbf legs. Only s3gm
# is still guarded off.
#
# Geo-FNO matters most here. It is the accuracy leader on this regime (0.385
# rel-L2 at 1%) and the 2D spectra figure does not contain it, so the figure
# currently cannot say whether the most accurate method is also the most
# spectrally faithful -- which is the distortion-perception question the paper
# raises. Same canonical frame, sensors and seed as every other panel:
# val frame 256, n_obs 655, seed 0, fingerprint sensors=655 idx_sum=22128955.
set -uo pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
WT=/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd "$WT/src"
KOLM=$WT/Save_TrainedModel/kolmogorov2d
OUT=$WT/Save_TrainedModel_pof/field_dumps
mkdir -p "$OUT"
FAIL=0
run_dump () {
  local TAG=$1; shift
  if [ -s "$OUT/$TAG.npz" ]; then echo "##### $TAG present, skip"; return 0; fi
  echo "##### $TAG start $(date +%T)"
  if python eval_kolm_ensemble.py "$@" --dump-npz "$OUT/$TAG.npz"; then
    echo "##### $TAG OK"; else echo "##### $TAG FAILED rc=$?"; FAIL=1; fi
}
run_dump kolm_geofno --model geofno \
  --run-dir "$(ls -d $KOLM/baseline_geofno/Baseline_geofno_Stage1_DemoN65_* | tail -1)" \
  --ckpt best --dump-frame 256 --n-obs-list 655 --seed 0
run_dump kolm_mlprbf --model mlp_rbf \
  --run-dir "$(ls -d $KOLM/baseline_mlp_rbf/Baseline_mlp_rbf_Stage1_DemoN64_* | tail -1)" \
  --ckpt best --dump-frame 256 --n-obs-list 655 --seed 0
echo "=== done; dumps present:"; ls -la "$OUT" | grep kolm_
exit $FAIL
