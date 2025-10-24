#!/bin/bash
#SBATCH --job-name=runtimes_case500_analysis
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=%x-%j.out

julia scripts/main_runtimes.jl case500 true
