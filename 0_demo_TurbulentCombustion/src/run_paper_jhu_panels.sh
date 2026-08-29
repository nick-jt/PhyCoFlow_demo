#!/bin/bash
#SBATCH --job-name=jhu_panels
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=120G

# ==========================================================================
# LEGACY QUARANTINE (baseline audit 2026-08-29): this launcher predates the
# canonical sensor-draw protocol (helpers.build_sparse_condition under
# torch.manual_seed(seed*777+snap) on an H100 SXM compute node; fingerprint
# snap=29 sensors=39062 idx_sum=37987162596). Its numbers are NOT
# comparable to canonical results and must not enter the paper table.
echo "WARNING: LEGACY NON-CANONICAL SENSOR DRAW -- numbers not comparable to the canonical fingerprint. Set ALLOW_LEGACY_EVAL=1 to run."
[ -z "${ALLOW_LEGACY_EVAL:-}" ] && exit 1
# ==========================================================================
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
CFG=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config
FIG=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Paper/iclr2027/figures
LFM=$(readlink -f "$(ls -d ../Save_TrainedModel/JHU/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN23_* | tail -1)")
L=jhu_panels_${SLURM_JOB_ID}.log
# Step 1: dump latent-FM's K=25 ensemble using the baseline's OWN sampler.
for S in 0 12; do
  echo "##### latent_fm ensemble dump, snapshot $S" >> $L
  ENSEMBLE_K=25 ENSEMBLE_NPZ=$FIG/ens_latentfm_s${S}.npz \
  python evaluate_Gen_Baseline.py --config "$LFM/run_config.yaml" \
      --run-dir "$LFM" --training-stage 2 --split val --snapshot-index $S 2>&1 \
    | grep -aE "\[ensemble\]|Traceback|Error|Exception|error:" >> $L
done
# Step 2: build the panels (ours sampled inline, latent-FM read from the dumps).
for S in 0 12; do
  echo "##### panels, snapshot $S" >> $L
  python paper_jhu_panels.py --snapshot $S --K 25 --nfe 4 >> $L 2>&1 \
    || echo "  [snapshot $S FAILED]" >> $L
done
echo "ALL DONE" >> $L
