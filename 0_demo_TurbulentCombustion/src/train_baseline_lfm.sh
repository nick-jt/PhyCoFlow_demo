#!/bin/bash
#SBATCH --job-name=bsln_lfm
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=72G

set -u

source ~/envs/jhtdb

DEMO_NUM=0
CONFIG_PATH="Save_config/config_baseline_Gen.yaml"

echo "Current directory:"
echo $(pwd)

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

echo "Running LFM stage 1 training on GPU 0..."
LOG_FILE="train_baseline_lfm_stage1_${SLURM_JOB_ID}_DemoN${DEMO_NUM}.log"
CUDA_VISIBLE_DEVICES=0 python train_Gen_Baseline.py \
    --config "$CONFIG_PATH" \
    --training-stage 1 \
    > "$LOG_FILE" 2>&1
lfm_stage1_status=$?

echo "LFM stage 1 exit status: $lfm_stage1_status"

if [[ "$lfm_stage1_status" -ne 0 ]]; then
    echo "LFM stage 1 training failed. See ${LOG_FILE} for details."
    exit 1
fi

echo "Running LFM stage 2 training on GPU 0..."
LOG_FILE="train_baseline_lfm_stage2_${SLURM_JOB_ID}_DemoN${DEMO_NUM}.log"
CUDA_VISIBLE_DEVICES=0 python train_Gen_Baseline.py \
    --config "$CONFIG_PATH" \
    --training-stage 2 \
    > "$LOG_FILE" 2>&1
lfm_stage2_status=$?

echo "LFM stage 2 exit status: $lfm_stage2_status"

if [[ "$lfm_stage2_status" -ne 0 ]]; then
    echo "LFM stage 2 training failed. See ${LOG_FILE} for details."
    exit 1
fi

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
