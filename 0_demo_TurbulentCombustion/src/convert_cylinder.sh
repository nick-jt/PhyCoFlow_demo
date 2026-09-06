#!/bin/bash
#SBATCH --job-name=conv_cyl2d
#SBATCH --partition=shared
#SBATCH --account=ammoniacomb
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=conv_cyl2d_%j.log

set -euo pipefail
source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"
python3 convert_cylinder.py --verify
echo "conversion done"
