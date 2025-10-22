#!/bin/bash
#SBATCH --job-name=runtimes_case2000_warm
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=%x-%j.out

julia main_runtimes.jl case2000 false
