#!/bin/bash
#SBATCH --job-name=lq_gtg    # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=875532@stud.unive.it     # Where to send mail	
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=168:00:00               # Time limit hrs:min:sec
#SBATCH --output=empty_log.log   # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1      # number of GPUs per node
#SBATCH --gpus-per-task=1      # number of GPUs per process


conda activate al_env

python3 app/main.py -m lq_gtg -ds cifar10 -tr 1 -s mixed -ts mean -plb 0.25 --no_none_strategy
#python3 app/main.py -m lq_gtg -ds cifar10 -tr 1 -s mixed -plb 0.1
#python3 app/main.py -m random entropy ll gtg gtg_ll -ds cifar10 cifar100 svhn -tr 1 -s mixed -ts mean 
