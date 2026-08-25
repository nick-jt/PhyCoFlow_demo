#!/bin/bash
#SBATCH --job-name=eval_ffm_N2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=72G

set -u

source ~/envs/jhtdb


DEMO_NUM=2

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

echo "Running FFM evaluation on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python evaluate_ffm.py --Demo-Num ${DEMO_NUM} --demo-root .. --snapshot-index 60 --extra-metrics ssim grad spectrum

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
