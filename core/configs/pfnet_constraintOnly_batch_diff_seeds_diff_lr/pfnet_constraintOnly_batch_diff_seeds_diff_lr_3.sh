#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH -p cpu-gpu-rtx8000
#SBATCH --gpus=1

python main.py --config pfnet_constraintOnly_batch_diff_seeds_diff_lr/pfnet_constraintOnly_batch_diff_seeds_diff_lr_3