#!/bin/bash
#SBATCH --job-name=C10_AL    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=875532@stud.unive.it     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=app_log.log   # Standard output and error log
## #SBATCH --gres=gpu:1
## #SBATCH --gpus-per-task=1       # number of GPUs per process
## #SBATCH --gpu-bind=single:1     # bind each process to its own GPU (single:<tasks_per_gpu>)
#SBATCH --gpus=tesla:2

echo "Running Training Cifar"

date

conda activate test

python3 app/main.py

date