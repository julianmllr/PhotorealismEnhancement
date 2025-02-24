#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --job-name=test_epe_kit

module load python
conda activate epe4

python epe/EPEExperiment.py test ./config/test_pfd2kit.yaml