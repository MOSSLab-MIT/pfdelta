#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH -p cpu-gpu-rtx8000
#SBATCH --gpus=1

python main.py --config canos_mse_batch_diff_seeds_diff_lr/canos_mse_batch_diff_seeds_diff_lr_0