#!/bin/bash

#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name="jiangm-job"
#SBATCH --mem=100G
#SBATCH --open-mode=append
#SBATCH --time=00:50:00
#SBATCH --partition=batch
#SBATCH --mem=50G

srun python profiling.py
