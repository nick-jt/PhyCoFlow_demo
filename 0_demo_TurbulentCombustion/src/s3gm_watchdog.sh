#!/bin/bash
#SBATCH --job-name=s3gm_wdog
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard
#SBATCH --account=f2pde
#SBATCH --mem=8G
set -u
source ~/envs/jhtdb
cd /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src
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
