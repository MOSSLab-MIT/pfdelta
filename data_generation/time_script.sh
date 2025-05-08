#!/bin/bash
#SBATCH --job-name=n2_e_57
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --output=%x-%j.out

source ~/.bashrc

srun julia -p 23 main.jl parallel case57 n-2
# julia main.jl linear case14 n-2