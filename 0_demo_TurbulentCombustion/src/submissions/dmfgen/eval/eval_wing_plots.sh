#!/bin/bash
#SBATCH --job-name=wing_plots
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=96G
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
W=/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/Save_TrainedModel/wing/pointcloud_ffm
L=eval_wing_plots_${SLURM_JOB_ID}.log
for R in iclr_wing_v3_expanded_DemoN13_20260818_084059 iclr_wing_v4_sym_DemoN19_20260819_083259; do
  for NS in 2 4; do
    echo "##### $R n_steps=$NS" >> $L
    mkdir -p $W/$R/Evaluation
    python evaluate_wing.py --run-dir $W/$R --ckpt best.pt \
        --K 4 --n-steps $NS --n-taps 512 --n-shear 128 --n-cases 8 --plots \
        --out $W/$R/Evaluation/wing_eval_nfe${NS}.json >> $L 2>&1
    echo "exit=$?" >> $L
  done
done
echo "ALL DONE" >> $L
