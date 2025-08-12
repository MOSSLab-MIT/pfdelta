#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH -p cpu-gpu-rtx8000
#SBATCH --gpus=1

python main.py --config canos_combined_loss_batch_diff_seeds_5e-4/canos_combined_loss_batch_diff_seeds_5e-4_1