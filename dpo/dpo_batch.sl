#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
#SBATCH -t 2-
#SBATCH -p a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

module purge
module load anaconda
conda activate dpo
cd /path/to/work/folder
accelerate launch train_dpo.py