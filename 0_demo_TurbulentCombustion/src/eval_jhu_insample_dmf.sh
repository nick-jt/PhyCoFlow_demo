#!/bin/bash
#SBATCH --job-name=jhu_insample_dmf
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=160G
#SBATCH --output=jhu_insample_%j.log
#
# 3D IN-SAMPLE LEAKAGE PROBE (2026-09-14). The frozen canonical JHU checkpoints
# are scored on 50 frames drawn from the TRAIN cubes (0-2) instead of the
# held-out cube 3 -- the 3D analogue of the 2D seen-trajectory probe in
# eval_kolm_litprotocol.py, and the ceiling any random-in-time split could
# reach (a training frame is r = 1.00 with itself). IDW / nearest neighbour on
# the same frames are the training-free frame-difficulty control.
#
# Same protocol as the canonical rows in every other respect: seed 0, sensors
# 19531 per observed field on cond_fields [0, 2], K = 8, NFE 4, snapshot
# positions chosen by np.random.default_rng(0).choice(150, 50) in every leg, and
# sensors drawn under torch.manual_seed(seed*777 + position). Outputs are keyed
# 'insample' in the filename and carry split / protocol / snapshot_ids in the
# payload, so they can never be selected as canonical rows.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

# Absolute: eval_latentfm_ensemble resolves a relative --run-dir against the
# repo directory, not src/, so ../Save_TrainedModel would escape the repo.
J=/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/Save_TrainedModel/JHU
DATA=/work/hdd/bilr/ntricard/datasets/JHU_4cubes_stride100.h5
LFM=$J/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN24_20260828_164541
DMF=$J/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100
FNO=$J/baseline_fno/fno3d_matched_DemoN90_20260828_181849
RC=0

# DMF-Gen leg only: 3151158 finished the other three legs and timed out here.
echo "=== DMF-Gen (best) in-sample ==="
python -u ensemble_eval.py --run-dir "$DMF" --ckpt best.pt \
  --K 8 --n-steps 4 --n-snapshots 50 --n-obs 19531 19531 --cond-fields 0 2 \
  --seed 0 --op-seed 1000 --chunk 1953125 --no-figs --split train \
  --out "$DMF/Evaluation/insample_train_all50_nfe4_K8.json" || RC=1

echo "rc=$RC end=$(date)"
exit $RC
