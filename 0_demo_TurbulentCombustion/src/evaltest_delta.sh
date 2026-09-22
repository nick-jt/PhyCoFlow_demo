#!/bin/bash
#SBATCH --job-name=evaltest
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=96G
#SBATCH --output=slurm_logs/evaltest_%j.log
# Early exercise of eval_kolm_ensemble.py on Delta (points path on a COPY of a
# best-so-far checkpoint so no crps_snap*.json contaminates the real eval dir;
# surface path on the smoke runs). 3 frames each.
set -uo pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR/src
T=/work/hdd/bilr/ntricard/evaltest_runs; rm -rf $T; mkdir -p $T/kolm_senseiver
cp $SLURM_SUBMIT_DIR/Save_TrainedModel/kolmogorov2d/baseline_det/Baseline_senseiver_Stage1_DemoN61_20260908_125038/best.pt $SLURM_SUBMIT_DIR/Save_TrainedModel/kolmogorov2d/baseline_det/Baseline_senseiver_Stage1_DemoN61_20260908_125038/run_config.yaml $SLURM_SUBMIT_DIR/Save_TrainedModel/kolmogorov2d/baseline_det/Baseline_senseiver_Stage1_DemoN61_20260908_125038/dataset_stats.pt $T/kolm_senseiver/ 2>/dev/null || cp $SLURM_SUBMIT_DIR/Save_TrainedModel/kolmogorov2d/baseline_det/Baseline_senseiver_Stage1_DemoN61_20260908_125038/*.yaml $T/kolm_senseiver/
echo "=== [A] senseiver kolm (points) ==="
python eval_kolm_ensemble.py --model senseiver --run-dir $T/kolm_senseiver --ckpt best --K 8 --n-obs-list 655 --cond-fields 0 --expect-val-len 640 --stratify-blocks 1 --out-prefix evaltest --n-frames 4 --seed 0 --op-seed 1000 --no-figs || echo "[A] FAILED"
echo "=== [B] senseiver cyl SURFACE smoke run ==="
python eval_kolm_ensemble.py --model senseiver --run-dir $SLURM_SUBMIT_DIR/Save_TrainedModel/_legacy/smoke_cyl_surface/baseline_det/Baseline_senseiver_Stage1_DemoN992_20260908_131451 --ckpt best --K 8 --n-obs-list 32 360 --cond-fields 0 1 2 --cond-source surface --expect-val-len 600 --stratify-blocks 2 --out-prefix evaltest_surface --n-frames 4 --seed 0 --op-seed 1000 --no-figs || echo "[B] FAILED"
echo "=== [C] dmfgen cyl SURFACE smoke run ==="
python eval_kolm_ensemble.py --model dmfgen --run-dir $SLURM_SUBMIT_DIR/Save_TrainedModel/_legacy/smoke_cyl_surface_ffm_DemoN993_20260908_130536 --ckpt best --K 8 --nfe 4 --n-obs-list 32 360 --cond-fields 0 1 2 --cond-source surface --expect-val-len 600 --stratify-blocks 2 --out-prefix evaltest_surface --n-frames 4 --seed 0 --op-seed 1000 --no-figs || echo "[C] FAILED"
echo "=== outputs ==="; ls $SLURM_SUBMIT_DIR/Save_TrainedModel/_legacy/smoke_cyl_surface/baseline_det/Baseline_senseiver_Stage1_DemoN992_20260908_131451/Evaluation $SLURM_SUBMIT_DIR/Save_TrainedModel/_legacy/smoke_cyl_surface_ffm_DemoN993_20260908_130536/Evaluation $T/kolm_senseiver/Evaluation 2>/dev/null
echo "=== evaltest done ==="
