#!/bin/bash
#SBATCH --job-name=n_u_2000
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=30
#SBATCH --output=%x-%j.out
#SBATCH --nodelist=node21


source ~/.bashrc

srun julia -p 29 main.jl uniform_parallel case2000 none
srun julia -p 29 main.jl uniform_parallel case2000 none
srun julia -p 29 main.jl uniform_parallel case2000 none
srun julia -p 29 main.jl uniform_parallel case2000 none
srun julia -p 29 main.jl uniform_parallel case2000 none
srun julia -p 29 main.jl uniform_parallel case2000 none
srun julia -p 29 main.jl uniform_parallel case2000 none
srun julia -p 29 main.jl uniform_parallel case2000 none
srun julia -p 29 main.jl uniform_parallel case2000 none
srun julia -p 29 main.jl uniform_parallel case2000 none


sbatch script_2000_n.sh