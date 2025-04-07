#!/bin/bash
#SBATCH --job-name=diffomer       # Job name
#SBATCH --output=log_slurm/result_cuongdm.txt      # Output file
#SBATCH --error=log_slurm/error_cuongdm.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=20                              # Number of CPU cores per task
sh bash-1.sh
