#!/bin/bash
#SBATCH --job-name=wing_base
# 24h is the hard cap on gpu-h100-stdby, where this account's jobs currently
# route. Each config sets shared.reload: true and checkpoints every 25 epochs,
# so a preempted or timed-out job resumes from last.pt on resubmission.
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --output=train_wing_%x_%j.log
#
# Full training for ONE SHIFT-WING learned baseline, conditioned on the
# body-surface observation pool. Submitted by the fleet with an afterok
# dependency on smoke_wing_baselines.sh.
#
#   MODEL=senseiver sbatch --job-name=wing_senseiver train_wing_baseline.sh
#   MODEL=mlp_rbf   sbatch --job-name=wing_mlprbf    train_wing_baseline.sh
#   MODEL=sit       sbatch --job-name=wing_sit       train_wing_baseline.sh
#
# Budget (all three): batch_size 8 x 75 steps/epoch x 4000 epochs =
# 300,000 optimizer steps, matched to iclr_wing_v3_expanded_DemoN13. Each
# config carries the full fairness contract in its header.
#
# No trailing `echo "exit status: $?"`: that pattern forces exit 0 and makes a
# crashed job report COMPLETED.

set -euo pipefail

# DELIBERATELY UNSET. train_iclr_wing_v3.sh does not export WING_AUGMENT, so
# our model trained WITHOUT spanwise reflection; exporting it here would train
# the baseline on a different data distribution. (The superseded
# train_sit_wing.sh stub did export it.)
unset WING_AUGMENT || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/envs/jhtdb

WT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion
cd "$WT/src"

MODEL=${MODEL:?set MODEL=senseiver|mlp_rbf|sit}
case $MODEL in
  senseiver) CFG=config_baseline_wing_senseiver.yaml; DRIVER=train_Det_Baseline.py ;;
  mlp_rbf)   CFG=config_baseline_wing_mlprbf.yaml;    DRIVER=train_Det_Baseline.py ;;
  sit)       CFG=config_baseline_SiT_wing.yaml;       DRIVER=train_Gen_Baseline.py ;;
  *) echo "[launcher] unknown MODEL=$MODEL"; exit 1 ;;
esac

echo "=== node $(hostname) job ${SLURM_JOB_ID} MODEL=$MODEL CFG=$CFG ==="
echo "WING_AUGMENT=${WING_AUGMENT:-<unset, as intended>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "$DRIVER" --config "$WT/Save_config/$CFG" \
    --training-stage 1 --device cuda:0
