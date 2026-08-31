#!/bin/bash
#SBATCH --job-name=sit_matched
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
# Capacity-matched SiT-point: depth 6 x hidden 240 x 4 heads = 6,571,924 params
# (+1.01% of the 6,506,253-param comparison model). The older
# config_baseline_SiT_xcube.yaml (hidden 256) is 7,468,804 = +14.8%, outside
# the +/-10% band.
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Measured: 12.85 -> 11.96 s/epoch (-6.9%). DataLoader workers are otherwise
# respawned every epoch. Node-local staging was measured and rejected: the H5
# reads at 1.27 GB/s and /tmp is no faster than Lustre here (5.9 s vs 5.9 s).
export JHU_PERSISTENT_WORKERS=1
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
CFGROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config
LOG_FILE="train_sit_matched_${SLURM_JOB_ID}.log"
# NOTE: deliberately no trailing `echo "exit status: $?"` -- that pattern makes
# the batch script exit 0 and a crashed run reports COMPLETED to sacct.
CUDA_VISIBLE_DEVICES=0 python train_Gen_Baseline.py \
        --config $CFGROOT/config_baseline_SiT_xcube_matched.yaml \
        --training-stage 1 >> "$LOG_FILE" 2>&1
RC=$?
echo "train rc=$RC" >> "$LOG_FILE"
if [ "$RC" -ne 0 ]; then exit "$RC"; fi

# Cost instrumentation on the run that was just produced (writes
# <run_dir>/instrumentation.json and prints greppable [instr] lines).
RUN=$(readlink -f "$(ls -d ../Save_TrainedModel/JHU/baseline_sit/matched/Baseline_sit_Stage1_DemoN41_* | tail -1)")
echo "RUN=$RUN" >> "$LOG_FILE"
for CK in best last; do
  python instrument_sit.py --run-dir "$RUN" --checkpoint $CK \
    --out "$RUN/instrumentation_${CK}.json" >> "$LOG_FILE" 2>&1
done
