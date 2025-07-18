#!/bin/bash
#SBATCH --job-name=n_u_2000
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=72
#SBATCH --output=%x-%j.out

source ~/.bashrc

srun julia -p 71 main.jl uniform_parallel case2000 none
# julia main.jl linear case14 n-2