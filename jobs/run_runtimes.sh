#!/bin/bash
#SBATCH --job-name=runtimes_57
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=%x-%j.out

julia main_runtimes.jl case57 false
