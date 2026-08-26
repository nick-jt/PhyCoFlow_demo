#!/bin/bash
#SBATCH --job-name=n31_ops
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=10
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
STM=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel
N31=$(ls -d $STM/firebench/pointcloud_ffm/iclr_firebench_v5clean_DemoN31_* | tail -1)
L=eval_n31_ops_${SLURM_JOB_ID}.log
echo "N31=$N31" >> $L
mkdir -p "$N31/Evaluation"
run_ours () {  # $1 op name  $2.. extra args
  OP=$1; shift
  echo "##### N31 $OP" >> $L
  python ensemble_eval.py --run-dir "$N31" --ckpt best.pt --K 8 --n-steps 4 \
      --n-snapshots 8 --op-seed 1000 "$@" \
      --out "$N31/Evaluation/ops_${OP}_K8.json" >> $L 2>&1
}
run_ours clean
run_ours noise01 --noise-sigma 0.1
run_ours noise03 --noise-sigma 0.3
run_ours slab25 --occlude slab:0.25
run_ours dropvw --drop-fields 1 2
echo "ALL DONE" >> $L
