#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH -p cpu-gpu-rtx8000
#SBATCH --gpus=1

python main.py --config graphconv_test_varySeedsLR/graphconv_test_varySeedsLR_6