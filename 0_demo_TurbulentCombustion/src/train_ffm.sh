#!/bin/bash
#SBATCH --job-name=ffm_uniform
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mail-user=ntricard@mit.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=72G

set -u

source ~/envs/jhtdb


DEMO_NUM=3

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU count: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"
echo "  GPU 0: $(CUDA_VISIBLE_DEVICES=0 python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo ""
echo "=========================================="

echo "Running FFM training on GPU 0..."
LOG_FILE="train_pointcloud_ffm_${SLURM_JOB_ID}_DemoN${DEMO_NUM}.log"
CUDA_VISIBLE_DEVICES=0 python train_pointcloud_ffm.py \
        --RELOAD \
        --config /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config/pointcloud_ffm/config_pointcloud_ffm_DemoN3_20260713_081738.yaml \
        --Demo-Num $DEMO_NUM >> "$LOG_FILE" 2>&1
ffm_status=$?

echo "FFM exit status: $ffm_status"

if [[ "$ffm_status" -ne 0 ]]; then
    echo "FFM training failed. See ${LOG_FILE} for details."
    exit 1
fi

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
