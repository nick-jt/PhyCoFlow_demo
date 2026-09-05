#!/bin/bash
#SBATCH --job-name=classical_kolm_sweep
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=32G
#SBATCH --output=classical_kolm_sweep_%j.log

# Sensor-count sweep for the performance-vs-N figure (2D panel).
# N = {65,164,655,1965,6554} = 0.1/0.25/1/3/10% of 256^2.
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

python -u baseline_classical_2d.py \
    --h5 /projects/ammoniacomb/generative_reconstruction/kolmogorov2d/Kolmogorov2D_shu_stride4.h5 \
    --train-frames 2560 --fields vorticity --cond-fields 0 \
    --methods constant kdtree idw gappy_pod --pod-rank 80 --idw-k 8 \
    --n-sensors 65 164 655 1965 6554 \
    --no-periodic --tag sweep_nonperiodic \
    --out-dir ../Save_TrainedModel/kolmogorov2d/baseline_classical
echo "sweep done"
