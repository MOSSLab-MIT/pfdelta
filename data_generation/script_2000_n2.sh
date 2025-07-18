#!/bin/bash
#SBATCH --job-name=n2_u_2000
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=70
#SBATCH --output=%x-%j.out
#SBATCH --nodelist=node28

source ~/.bashrc

srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2
srun julia -p 65 main.jl uniform_parallel case2000 n-2


sbatch script_2000_n2.sh