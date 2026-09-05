#!/bin/bash
#SBATCH --job-name=dump_fields_pof
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# Per-method field dumps on the fixed qualitative-figure snapshot (val idx 3),
# sensor draw matched bit-for-bit to qual_jhu.npz / qual_firebench.npz.
# Each dump is INDEPENDENT: a partial run (standby preemption / wall) still
# leaves every completed npz usable. Ordered cheapest-first; SiT-point
# (K=8 x ~2.5 min/sample) runs last.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb

WT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion
MAIN=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
STM=$MAIN/Save_TrainedModel
FIG=$MAIN/Paper/iclr2027/figures
OUT=$WT/Save_TrainedModel_pof/field_dumps
mkdir -p "$OUT"
cd "$WT/src"
L=$WT/src/dump_fields_pof_${SLURM_JOB_ID}.log

# Paper-table checkpoints:
#   SiT-point  : baseline_sit/matched (assemble_baseline_table.py "SiT-point")
#   Senseiver  : DemoN43 (assemble_baseline_table.py "Senseiver")
#   FB latentFM: Stage2 DemoN35 (eval_firebench_ops.sh LFM)
#   FB Senseiver: DemoN36 (eval_firebench_ops.sh SEN, the dir with best.pt)
SIT_JHU=$STM/JHU/baseline_sit/matched/Baseline_sit_Stage1_DemoN41_20260828_181938
SEN_JHU=$STM/JHU/baseline_senseiver/Baseline_senseiver_Stage1_DemoN43_20260828_180959
LFM_FB=$STM/firebench/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN35_20260823_234826
SEN_FB=$STM/firebench/baseline_senseiver/Baseline_senseiver_Stage1_DemoN36_20260823_153857

run_dump () {  # $1 tag, rest: args
  TAG=$1; shift
  if [ -s "$OUT/$TAG.npz" ]; then
    echo "##### $TAG already present, skipping" >> "$L"; return 0
  fi
  echo "##### $TAG start $(date +%T)" >> "$L"
  python dump_fields_baseline.py "$@" --out "$OUT/$TAG.npz" >> "$L" 2>&1 \
    && echo "##### $TAG OK $(date +%T)" >> "$L" \
    || echo "##### $TAG FAILED rc=$? $(date +%T)" >> "$L"
}

# Snapshot pinning: qual_jhu.npz is val[3] of the ratio-0.75/gap-0 split =
# ABSOLUTE frame 153; qual_firebench.npz is val[3] of ours' ratio-0.9/gap-10
# split of the merged FB h5 = ABSOLUTE frame 112 (the FB baselines use
# ratio 0.75, where frame 112 is val index 22). --frame resolves this.
# DemoN43 (JHU senseiver) trained from a node-local /tmp staging copy of the
# H5 that no longer exists -> --data-path points at the shared original.
JHU_H5=/projects/ammoniacomb/generative_reconstruction/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5

# --- JHU (split env exactly as run_qualitative.sh / seeded JHU evals) --------
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
run_dump jhu_senseiver --method senseiver --run-dir "$SEN_JHU" --ckpt best \
  --split val --frame 153 --sensor-seed 103 --cond-fields 0 2 --n-obs 19531 \
  --data-path "$JHU_H5" --check-qual "$FIG/qual_jhu.npz"

# --- FireBench (split env exactly as run_qualitative_fb.sh / fb_ops evals) ---
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=10
run_dump firebench_senseiver --method senseiver --run-dir "$SEN_FB" --ckpt best \
  --split val --frame 112 --sensor-seed 1003 --cond-fields 0 1 2 --n-obs 36772 \
  --field-names u v w theta rho_f --check-qual "$FIG/qual_firebench.npz"

run_dump firebench_latent_fm --method latent_fm --run-dir "$LFM_FB" --ckpt best \
  --split val --frame 112 --sensor-seed 1003 --cond-fields 0 1 2 --n-obs 36772 \
  --K 8 --n-steps 4 --field-names u v w theta rho_f \
  --check-qual "$FIG/qual_firebench.npz"

# --- JHU SiT-point last (most expensive) -------------------------------------
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
run_dump jhu_sit --method sit --run-dir "$SIT_JHU" --ckpt best \
  --split val --frame 153 --sensor-seed 103 --cond-fields 0 2 --n-obs 19531 \
  --K 8 --check-qual "$FIG/qual_jhu.npz"

echo "ALL DONE $(date +%T); files:" >> "$L"
ls -la "$OUT" >> "$L" 2>&1
