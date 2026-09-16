#!/bin/bash
#SBATCH --job-name=s3gm_wdog
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=8G
set -u
source ~/envs/jhtdb
cd /work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/src
TRAIN_JOB=${TRAIN_JOB:?set TRAIN_JOB}
L="s3gm_watchdog_${SLURM_JOB_ID}.log"
python -u s3gm_watchdog.py \
    --log "s3gm_train_${TRAIN_JOB}.log" \
    --job-id "$TRAIN_JOB" \
    --out "s3gm_verdict_${TRAIN_JOB}.json" \
    --gate-steps 15000 --loss-max 0.30 \
    --poll-s 120 >> "$L" 2>&1
RC=$?
echo "watchdog rc=$RC" >> "$L"
exit $RC
