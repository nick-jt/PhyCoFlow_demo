#!/bin/bash
#SBATCH --job-name=kolm_litproto
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --output=eval_kolm_litproto_%j.log

# Split-design ablation: literature-style "seen trajectory, unseen frame"
# protocol + uniform-lattice sensor arm, DMF-Gen vs IDW k=8. See
# eval_kolm_litprotocol.py's docstring. Outputs land in
# Save_TrainedModel/kolmogorov2d/litprotocol/.
#
# NOTE: deliberately no trailing `echo "exit status: $?"` -- that pattern
# forces exit 0 and makes a crashed job report COMPLETED.

set -uo pipefail

export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/envs/jhtdb

WT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion
cd "$WT/src"

DMF=$(readlink -f "$(ls -d $WT/Save_TrainedModel/kolmogorov2d/pointcloud_ffm/bench_kolm_v1_DemoN101_* | tail -1)")
echo "=== node $(hostname) job ${SLURM_JOB_ID} DMF=$DMF ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

python eval_kolm_litprotocol.py --run-dir "$DMF" \
  --K 8 --nfe 4 --n-obs 655 --idw-k 8 --n-frames 50 --seed 0 --fig-every 10
RC=$?

echo "=== outputs ==="
ls -la "$WT/Save_TrainedModel/kolmogorov2d/litprotocol/" 2>/dev/null || true
exit $RC
