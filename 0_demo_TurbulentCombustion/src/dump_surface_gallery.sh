#!/bin/bash
#SBATCH --job-name=surf_gallery
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=96G
#SBATCH --output=surf_gallery_%j.log
# Field dumps for the SURFACE-TO-FIELD gallery: same canonical frame as the
# volume-sensing cylinder gallery (val 300 = held-out Re 250), but conditioned
# only on the wall ring (--cond-source surface, all three fields at the taps,
# 128 taps/field). Lets the two galleries be read side by side: identical flow,
# identical frame, only the sensing geometry differs.
set -uo pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
STM=../Save_TrainedModel/cylinder2d_surface
OUT=../Save_TrainedModel_pof/field_dumps
mkdir -p $OUT
FAIL=0
dump () {  # dump <tag> <model> <run-glob> [extra]
  local TAG=$1 M=$2 G=$3; shift 3
  [ -s "$OUT/$TAG.npz" ] && { echo "##### $TAG present, skipping"; return 0; }
  local RUN; RUN=$(readlink -f "$(ls -d $G 2>/dev/null | tail -1)" 2>/dev/null)
  [ -z "$RUN" ] && { echo "##### $TAG: no run dir"; FAIL=1; return; }
  echo "##### $TAG <- $(basename $RUN)"
  python eval_kolm_ensemble.py --model "$M" --run-dir "$RUN" --ckpt best \
      --dump-frame 300 --dump-npz "$OUT/$TAG.npz" --n-obs-list 128 \
      --cond-fields 0 1 2 --cond-source surface --expect-val-len 600 \
      --stratify-blocks 2 --K 8 "$@" || { echo "##### $TAG FAILED"; FAIL=1; }
}
dump cylsurf_senseiver senseiver "$STM/baseline_det/Baseline_senseiver_Stage1_DemoN81_*"
dump cylsurf_geofno    geofno    "$STM/baseline_geofno/Baseline_geofno_Stage1_DemoN85_*"
dump cylsurf_dmfgen    dmfgen    "$STM/pointcloud_ffm/bench_cyl_surface_v1_DemoN103_*" --nfe 4
dump cylsurf_sit       sit       "$STM/baseline_sit/Baseline_sit_Stage1_DemoN83_*"
echo "=== done ==="; ls -la "$OUT"/cylsurf_*.npz 2>/dev/null
exit $FAIL
