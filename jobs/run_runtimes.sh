#!/bin/bash
#SBATCH --job-name=runtimes_case57
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=%x-%j.out

julia scripts/main_runtimes.jl case57 true
