#!/usr/bin/env bash
#SBATCH -J create_dataset
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -o logs/create_dataset.%x.%A_%a.out
#SBATCH -e logs/create_dataset.%x.%A_%a.err
set -euo pipefail

module purge
module load julia/1.9.1

# # Pick argument by array index
# PERTS=(none)
# PERT="${PERTS[$SLURM_ARRAY_TASK_ID]}"
PERT="none"

cd "$HOME/pfdelta/data_generation"
julia main.jl linear case500 $PERT /home/akrivera/orcd/scratch/pfdelta_data
