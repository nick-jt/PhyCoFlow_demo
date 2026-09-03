#!/bin/bash
#SBATCH --job-name=conv_kolm2d
#SBATCH --partition=short
#SBATCH --account=ammoniacomb
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=conv_kolm2d_%j.log

set -euo pipefail
cd "$(dirname "$0")"
python3 convert_kolmogorov.py --verify "$@"
