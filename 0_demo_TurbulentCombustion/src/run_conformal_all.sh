#!/bin/bash
#SBATCH --job-name=conformal
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=200G
#SBATCH --output=conformal_%j.log
# Route-2 conformalized recalibration on every density that dump_calib_points.sh
# produced. Fits the per-distance-bin conformal quantile on TUNE (odd snapshots),
# reports coverage frozen on TEST (even) before and after. Compute node: the
# dumps are O(GB) of npz.
#
# The output name must key the MODEL, not just the density tag: the tag
# (calib_points_n19531_K8_nfe4) is identical for every model, so a second model
# run with the default prefix would silently overwrite the first one's JSON.
# DMF-Gen keeps the historical prefix "conformal_"; any other run dir must
# pass its own, e.g.
#   RD=../Save_TrainedModel/JHU/baseline_latent_fm/<run> PREFIX=conformal_lfm_ sbatch run_conformal_all.sh
set -uo pipefail
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
DMFGEN_RD=../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100
RD=${RD:-$DMFGEN_RD}
PREFIX=${PREFIX:-conformal_}
if [ "$RD" != "$DMFGEN_RD" ] && [ "$PREFIX" = "conformal_" ]; then
  echo "[guard] non-DMF-Gen RD needs its own PREFIX (would overwrite DMF-Gen's conformal_*.json)"; exit 2
fi
OUT=../Paper/pof2026/recalib
mkdir -p $OUT
RC=0
for D in $RD/Evaluation/calib_points_n*_K8_nfe4; do
  [ -d "$D" ] || continue
  n=$(ls "$D"/calib_points_snap*.npz 2>/dev/null | wc -l)
  if [ "$n" -lt 40 ]; then echo "[skip] $(basename $D): only $n snapshots (dump incomplete)"; continue; fi
  tag=$(basename "$D")
  echo "=== $tag ($n snapshots) ==="
  python conformal_recalib.py --dump-dir "$D" --out "$OUT/${PREFIX}${tag}.json" || RC=1
done
exit $RC
