#!/bin/bash
#SBATCH --job-name=kolm_nfe_dump
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=48G
#SBATCH --output=kolm_nfe_dump_%j.log

# NFE-matched spectral control: the 2D spectra figure compares DMF-Gen at
# NFE 4 against SiT at 50 steps. Dump DMF-Gen on the SAME frame/sensors at
# NFE 16 and 50 so the small-scale deficit can be attributed to sampler
# budget or to architecture.
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

RUN=$(ls -d ../Save_TrainedModel/kolmogorov2d/pointcloud_ffm/bench_kolm_v1_DemoN101_* | tail -1)
OUT=../Save_TrainedModel_pof/field_dumps

for N in 16 50; do
    echo "=== NFE=$N ==="
    python -u eval_kolm_ensemble.py --model dmfgen --run-dir "$RUN" --ckpt best \
        --K 8 --nfe "$N" --n-obs-list 655 --seed 0 --op-seed 1000 \
        --dump-frame 256 --dump-npz "$OUT/kolm_dmfgen_nfe${N}.npz"
done
echo "nfe dump done"
