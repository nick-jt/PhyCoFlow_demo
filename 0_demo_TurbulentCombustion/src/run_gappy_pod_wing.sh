#!/bin/bash
#SBATCH --job-name=gappy_wing
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=standard
#SBATCH --account=f2pde
#SBATCH --mem=180G
# NOT YET LAUNCHED. CPU-only, no training required — can be run at any time.
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
W=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel/wing
mkdir -p $W/baseline_gappy_pod
for R in 32 64 128; do
  python baseline_gappy_pod_wing.py \
      --processed-root /projects/ammoniacomb/generative_reconstruction/shift_wing/processed_v3 \
      --rank $R --n-taps 512 \
      --out $W/baseline_gappy_pod/gappy_rank${R}.json >> gappy_pod_wing_${SLURM_JOB_ID}.log 2>&1
done
echo "ALL DONE" >> gappy_pod_wing_${SLURM_JOB_ID}.log
