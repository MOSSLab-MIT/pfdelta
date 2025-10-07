#!/bin/bash
#SBATCH --job-name=n1_2000_nochpt
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=72
#SBATCH --output=%x-%j.out

source ~/.bashrc

srun julia -p 71 main.jl parallel case2000 n-1
