#!/bin/bash
#SBATCH --job-name=n-2_2000
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=72
#SBATCH --output=%x-%j.out

source ~/.bashrc

srun julia -p 71 main.jl parallel case2000 n-2
