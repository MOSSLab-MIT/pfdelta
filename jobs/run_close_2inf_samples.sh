#!/usr/bin/env bash
#SBATCH -J close2inf
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -o logs/close2inf.%x.%j.out
#SBATCH -e logs/close2inf.%x.%j.err
set -euo pipefail

mkdir -p logs
module purge
module load julia/1.9.1
module load matlab/matlab-2024a

# Use repo env & run with config path
cd "$HOME/pfdelta"
CFG_PATH="${1:-configs/config_close2inf.toml}"
julia --project=. data_generation/create_close2infeasible.jl "$CFG_PATH"

echo "✅ close2inf completed at $(date)"
