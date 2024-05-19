#!/bin/bash
#SBATCH --job-name=exp_mod    # Job name
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=875532@stud.unive.it     # Where to send mail	
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # 2
#SBATCH --ntasks-per-node=1         # 2
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=300:00:00               # Time limit hrs:min:sec
#SBATCH --output=empty_log.log   # Standard output and error log
#SBATCH --gpus=quadro_rtx_5000:1
# ##SBATCH --gres=gpu:1           # 2
# ##SBATCH --gpus-per-node=1      # 2, number of GPUs per node
# ##SBATCH --gpus-per-task=1      # number of GPUs per process

conda activate al_env

python3 app/main.py -gpus 1 -m lq_gtg -ds cifar10 -tr 1 -am corr -am_ts mean -e_s mean -gtg_iter 10 #30
