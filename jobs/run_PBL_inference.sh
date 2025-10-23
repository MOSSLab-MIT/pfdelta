#!/bin/bash
#SBATCH -J pbl_${CASE}
#SBATCH -o pbl_%x_%j.out
#SBATCH -e pbl_%x_%j.err
#SBATCH -p cpu             # pick your CPU partition
#SBATCH -N 1
#SBATCH -c 4               # CPU cores
#SBATCH --mem=16G
#SBATCH -t 08:00:00        # walltime

set -euo pipefail

# --- repo root (adjust if needed)
cd "$HOME/pfdelta"

# --- conda setup (non-interactive)
eval "$(conda shell.bash hook)"
conda activate pfdelta_env


# Run (your script already aggregates across topo/sample_type/run)
python scripts/inference_results_PBL.py "${CASE}"

echo "[INFO] Done: $(date)"