#!/bin/bash
#SBATCH --job-name=n2_14
#SBATCH --time=4-00:00:00
#SBATCH --output=%x-%j.out


source ~/.bashrc

julia main.jl linear case14 n-2
