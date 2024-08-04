#!/bin/bash
#SBATCH --job-name=llmlp    # Job name
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=875532@stud.unive.it     # Where to send mail	
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # 2
#SBATCH --ntasks-per-node=1         # 2
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=500:00:00               # Time limit hrs:min:sec
#SBATCH --output=empty_log.log   # Standard output and error log
#SBATCH --gpus=quadro_rtx_5000:1

conda activate al_env

#python3 app/main.py -gpus 1 -m random entropy ll -ds cifar10 -tr 5 --temp_unlab_pool
#python3 app/main.py -gpus 1 -m llmlp_gtg lsmlps_gtg lstmreg_gtg lstmbc_gtg -ds cifar10 -tr 5 -am corr -am_ts none mean -e_s mean --temp_unlab_pool
python3 app/main.py -gpus 1 -m llmlp_gtg -ds cifar10 -tr 5 -am corr -am_ts none mean -e_s mean #--temp_unlab_pool