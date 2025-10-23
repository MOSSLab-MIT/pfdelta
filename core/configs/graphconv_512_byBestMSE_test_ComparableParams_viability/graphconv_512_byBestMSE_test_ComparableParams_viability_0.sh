#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH -p cpu-gpu-rtx8000
#SBATCH --gpus=1

python main.py --config graphconv_512_byBestMSE_test_ComparableParams_viability/graphconv_512_byBestMSE_test_ComparableParams_viability_0