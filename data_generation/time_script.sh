#!/bin/bash
#SBATCH --job-name=time_benchmark
#SBATCH --time=1-00:00:00


# #SBATCH --cpus-per-task=8

source ~/.bashrc

# srun julia -p 7 main.jl algorithm30
julia main.jl linear case14 n-2