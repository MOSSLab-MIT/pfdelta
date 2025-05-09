#!/bin/bash
#SBATCH --job-name=n2_u_118
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=30
#SBATCH --output=%x-%j.out

source ~/.bashrc

srun julia -p 29 main.jl uniform_parallel case118 n-2
# julia main.jl linear case14 n-2