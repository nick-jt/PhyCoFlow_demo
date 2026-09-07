#!/bin/bash
#SBATCH --job-name=kolm_ksweep
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G
#SBATCH --output=kolm_ksweep_%j.log

# DMF-Gen ensemble-size sweep on Kolmogorov: same 50 frames, same sensors,
# K in {16, 32, 64} (K=8 already on disk from job 17938333). Answers whether
# more posterior samples improve rel-L2(mean)/CRPS/coverage and by how much.
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

RUN=$(ls -d ../Save_TrainedModel/kolmogorov2d/pointcloud_ffm/bench_kolm_v1_DemoN101_* | tail -1)
for K in 16 32 64; do
    echo "=== K=$K ==="
    python -u eval_kolm_ensemble.py --model dmfgen --run-dir "$RUN" --ckpt best \
        --K "$K" --nfe 4 --n-obs-list 655 --n-frames 50 --seed 0 --op-seed 1000 \
        --fig-every 100
done
echo "ksweep done"
