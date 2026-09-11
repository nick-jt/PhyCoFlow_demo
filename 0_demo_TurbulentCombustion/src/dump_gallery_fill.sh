#!/bin/bash
#SBATCH --job-name=gallery_fill
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=120G
#SBATCH --output=gallery_fill_%j.log
# Field dumps for the all-baselines galleries, on the fleet's OWN canonical
# protocol so every panel is a reconstruction the tables actually scored.
#
# Cylinder, val frame 300 (canonical, si=25), 238 sensors per field on u_x,u_y,
# seed 0 -- exactly the fleet eval. Expected fingerprints (fleet JSONs):
#   mesh methods (mlp_rbf)                -> 476 sensors, idx_sum 5581141
#   grid methods (sit/latent_fm/geofno/s3gm) -> 476 sensors, idx_sum 19084851
# The old cyl_sit / cyl_latent_fm dumps used 800/field on the grid -- a draw
# the fleet never scored -- so they are moved to superseded_800obs/ first
# (moving a symlink moves the link, never its target; nothing is deleted).
# Kolmogorov, val frame 256, 655 sensors: S3GM only (everything else exists;
# expected idx_sum 22128955).
set -uo pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
WT=/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $WT/src
S=$WT/Save_TrainedModel; OUT=$WT/Save_TrainedModel_pof/field_dumps
mkdir -p $OUT/superseded_800obs
for f in cyl_sit.npz cyl_latent_fm.npz; do
  [ -e $OUT/$f ] && mv $OUT/$f $OUT/superseded_800obs/ && echo "moved $f -> superseded_800obs/"
done
FAIL=0
dump () { local TAG=$1; shift
  if [ -s $OUT/$TAG.npz ]; then echo "##### $TAG present, skip"; return; fi
  echo "##### $TAG start $(date +%T)"
  if python eval_kolm_ensemble.py "$@" --dump-npz $OUT/$TAG.npz; then echo "##### $TAG OK $(date +%T)"
  else echo "##### $TAG FAILED"; FAIL=1; fi; }
C="--ckpt best --dump-frame 300 --n-obs-list 238 --cond-fields 0 1 --expect-val-len 600 --K 8 --seed 0"
dump cyl_mlprbf    --model mlp_rbf   --run-dir $S/cylinder2d/baseline_mlp_rbf/Baseline_mlp_rbf_Stage1_DemoN72_20260906_082016 $C
dump cyl_geofno    --model geofno    --run-dir $S/cylinder2d/baseline_geofno/Baseline_geofno_Stage1_DemoN75_20260906_082017 $C
dump cyl_sit       --model sit       --run-dir $S/cylinder2d/baseline_sit/Baseline_sit_Stage1_DemoN73_20260906_082303 $C
dump cyl_latent_fm --model latent_fm --run-dir $S/cylinder2d/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN70_20260907_012015 $C --nfe 4
dump cyl_s3gm      --model s3gm      --run-dir $S/cylinder2d/baseline_s3gm/Baseline_s3gm_Stage1_DemoN74_20260906_082321 $C
dump kolm_s3gm     --model s3gm      --run-dir $S/kolmogorov2d/baseline_s3gm/Baseline_s3gm_Stage1_DemoN63_20260905_161350 \
     --ckpt best --dump-frame 256 --n-obs-list 655 --cond-fields 0 --expect-val-len 640 --K 8 --seed 0
echo "=== fingerprints ==="; grep -E "^\[seedcheck\]|#####" $WT/src/gallery_fill_${SLURM_JOB_ID}.log | grep -v start
exit $FAIL
