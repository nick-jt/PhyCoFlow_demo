#!/bin/bash
#SBATCH --job-name=fb_v5c_eng
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --mem=96G
# Ours on FireBench (config_iclr_firebench_v5clean, DemoN31) on MIT Engaging.
# Needs the merged file from merge_firebench_cases.py. Submit via
# submit_chain.sh; --RELOAD makes segments resume.
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=10 JHU_AUGMENT=reflect_y AUG_GRID_SHAPE=152,126,192
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
module load cuda/12.4.0
DEMO=/home/ntricard/projects/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $DEMO/src
DATA=${FB_DATA:-/home/ntricard/orcd/scratch/firebench3d/FireBench_u10u12_merged.h5}
L=$DEMO/src/train_fb_v5clean_eng_${SLURM_JOB_ID}.log
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date)" > $L

STAGE=/tmp/$USER/fb_${SLURM_JOB_ID}; mkdir -p $STAGE
SZ=$(stat -c %s $DATA); AVAIL=$(df -B1 --output=avail /tmp | tail -1); RUNDATA=$DATA
if [ "$AVAIL" -gt $((SZ + 20000000000)) ]; then
  cp $DATA $STAGE/ && DSZ=$(stat -c %s $STAGE/$(basename $DATA)) || DSZ=0
  if [ "$DSZ" = "$SZ" ]; then RUNDATA=$STAGE/$(basename $DATA); else rm -f $STAGE/$(basename $DATA); fi
fi
[ -f "${DATA%.h5}.stats.pt" ] && cp "${DATA%.h5}.stats.pt" "${RUNDATA%.h5}.stats.pt" 2>/dev/null
echo "data=$RUNDATA" >> $L

CFG=Save_config/fb_v5clean_eng.yaml
sed -e "s|^data:.*|data: \"$RUNDATA\"|" \
    $DEMO/Save_config/config_iclr_firebench_v5clean.yaml > $DEMO/$CFG

CUDA_VISIBLE_DEVICES=0 python -u train_pointcloud_ffm.py \
    --RELOAD --config $DEMO/$CFG --Demo-Num 31 >> $L 2>&1
RC=$?
[ -f "${RUNDATA%.h5}.stats.pt" ] && cp "${RUNDATA%.h5}.stats.pt" "${DATA%.h5}.stats.pt" 2>/dev/null
rm -rf $STAGE
echo "end=$(date) rc=$RC" >> $L
exit $RC
