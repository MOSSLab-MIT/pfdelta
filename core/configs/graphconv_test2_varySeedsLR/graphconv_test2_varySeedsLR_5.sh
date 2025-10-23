#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH -p cpu-gpu-rtx8000
#SBATCH --gpus=1

python main.py --config graphconv_test2_varySeedsLR/graphconv_test2_varySeedsLR_5