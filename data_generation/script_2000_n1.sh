#!/bin/bash
#SBATCH --job-name=n1_u_2000
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=60
#SBATCH --output=%x-%j.out
#SBATCH --nodelist=node15

source ~/.bashrc

srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1
srun julia -p 59 main.jl uniform_parallel case2000 n-1


sbatch script_2000_n1.sh