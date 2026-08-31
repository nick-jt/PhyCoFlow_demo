#!/bin/bash
#SBATCH --job-name=s3gm_final
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
# Post-training: (1) re-confirm the stabilised (alpha,beta) on the FINAL
# checkpoint, (2) evaluate the JHU-tuned arm, (3) evaluate the upstream arm to
# document its divergence. Both arms are reported; neither substitutes.
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
source ~/envs/jhtdb
cd /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src
RUN=$(ls -dt ../Save_TrainedModel/JHU/baseline_s3gm/matched/Baseline_s3gm_Stage1_DemoN94_* | head -1)
L="s3gm_finalize_${SLURM_JOB_ID}.log"
echo "RUN=$RUN" >> "$L"

echo "===== STEP 1: stability re-confirmation on FINAL checkpoint =====" >> "$L"
ISO_ARMS=v3 python -u s3gm_isolate.py --run-dir "$RUN" --ckpt last --n-steps 200 \
    --device auto --out "$RUN/isolation_final.json" >> "$L" 2>&1
echo "step1 rc=$?" >> "$L"

for CK in last best; do
  echo "===== STEP 2: eval arm=jhu_tuned ckpt=$CK =====" >> "$L"
  python -u eval_s3gm3d.py --run-dir "$RUN" --ckpt $CK --arm jhu_tuned \
      --seed 0 --op-seed 1000 --n-snapshots 50 --K 8 \
      --cond-fields 0 2 --n-obs 19531 19531 >> "$L" 2>&1
  echo "eval jhu_tuned $CK rc=$?" >> "$L"
done

echo "===== STEP 3: eval arm=upstream (documents divergence, 5 snaps) =====" >> "$L"
python -u eval_s3gm3d.py --run-dir "$RUN" --ckpt last --arm upstream \
    --seed 0 --op-seed 1000 --n-snapshots 5 --K 2 \
    --cond-fields 0 2 --n-obs 19531 19531 >> "$L" 2>&1
echo "eval upstream rc=$?" >> "$L"
