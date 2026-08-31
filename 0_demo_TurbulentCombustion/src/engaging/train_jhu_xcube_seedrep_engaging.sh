#!/bin/bash
#SBATCH --job-name=jhu_seedrep
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --mem=180G
# Training-seed replicates of canonical N29 (HANDOFF secondary item: "2-3
# seeds of N29; seed noise is the open rigor item"). spec02 config VERBATIM
# -- only seed and save_dir differ; Demo_Num stays 29 (same protocol; the
# canonical eval seed is separate and stays 0). Cross-cube 4-cube data,
# block split gap 0 (150 train = cubes 1-3, 50 eval = cube 4).
# Submit one 5-segment chain per seed:
#   SEED=1379 submit_chain.sh ... --export=ALL,SEED=1379
# Canonical evals of the finished replicates must run on the h100 node
# (fingerprint is H100-SXM-bound); training SKU is free.
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
module load cuda/12.4.0
DEMO=/home/ntricard/projects/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $DEMO/src
SEED=${SEED:?set SEED (e.g. 1379)}
DATA=${JHU_DATA:-$DEMO/Dataset/JHU_4cubes_stride100.h5}
L=$DEMO/src/train_jhu_seedrep${SEED}_eng_${SLURM_JOB_ID}.log
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) seed=$SEED start=$(date)" > $L

STAGE=/tmp/$USER/jhu4c_${SLURM_JOB_ID}; mkdir -p $STAGE
SZ=$(stat -c %s $DATA); AVAIL=$(df -B1 --output=avail /tmp | tail -1); RUNDATA=$DATA
if [ "$AVAIL" -gt $((SZ + 20000000000)) ]; then
  cp $DATA $STAGE/ && DSZ=$(stat -c %s $STAGE/$(basename $DATA)) || DSZ=0
  if [ "$DSZ" = "$SZ" ]; then RUNDATA=$STAGE/$(basename $DATA); else rm -f $STAGE/$(basename $DATA); fi
fi
[ -f "${DATA%.h5}.stats.pt" ] && cp "${DATA%.h5}.stats.pt" "${RUNDATA%.h5}.stats.pt" 2>/dev/null
echo "data=$RUNDATA" >> $L

CFG=Save_config/jhu_seedrep${SEED}_eng.yaml
sed -e "s|^data:.*|data: \"$RUNDATA\"|" \
    -e "s|^seed :.*|seed : $SEED|" \
    -e "s|^save_dir:.*|save_dir: \"Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_seedrep$SEED\"|" \
    $DEMO/Save_config/config_iclr_jhu_xcube_spec02.yaml > $DEMO/$CFG

CUDA_VISIBLE_DEVICES=0 python -u train_pointcloud_ffm.py \
    --RELOAD --config $DEMO/$CFG --Demo-Num 29 >> $L 2>&1
RC=$?
[ -f "${RUNDATA%.h5}.stats.pt" ] && cp "${RUNDATA%.h5}.stats.pt" "${DATA%.h5}.stats.pt" 2>/dev/null
rm -rf $STAGE
echo "end=$(date) rc=$RC" >> $L
exit $RC
