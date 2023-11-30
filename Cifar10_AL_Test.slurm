#!/bin/bash
#SBATCH --job-name=Cifar10_AL_test    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=875532@stud.unive.it     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=test2.log   # Standard output and error log
#SBATCH --gres=gpu:1

echo "Running Training Cifar"

date

conda activate test

python /app/main.py

date