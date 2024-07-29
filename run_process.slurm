#!/bin/bash
#SBATCH --job-name=AL_GTG    # Job name
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

#python3 app/main.py -gpus 1 -m ll_gtg -ds cifar10 svhn fmnist -tr 5 -am corr -am_ts mean -e_s mean -gtg_iter 30
#python3 app/main.py -gpus 1 -m ll_gtg -ds cifar10 -tr 5 -am corr -am_ts mean -e_s mean -gtg_iter 30
#python3 app/main.py -gpus 1 -m lstmbc_gtg -ds cifar10 cifar100 svhn fmnist -tr 1 -am corr -am_ts mean -e_s mean -gtg_iter 10
#python3 app/main.py -gpus 1 -m llmlp_gtg lsmlps_gtg lstmreg_gtg lstmbc_gtg -ds tinyimagenet -tr 1 -am corr -am_ts mean -e_s mean -gtg_iter 10


#python3 app/main.py -gpus 1 -m ra -ds cifar10 cifar100 svhn fmnist -tr 1 -am corr -am_ts mean -e_s mean -gtg_iter 30
#python3 app/main.py -gpus 1 -m ll_gtg -ds cifar100 -tr 4 -am corr -am_ts mean -e_s mean -gtg_iter 30WWWW
#python3 app/main.py -gpus 1 -m random tidal ll -ds tinyimagenet -tr 2

#python3 app/main.py -gpus 1 -m random entropy coreset ll tidal alphamix cdal badge bald tavaal -ds tinyimagenet -tr 5
#python3 app/main.py -gpus 1 -m tavaal -ds cifar10 -tr 5

#python3 app/main.py -gpus 1 -m llmlp_gtg -ds cifar10 -tr 5 -am corr -am_ts mean none -am_s mixed -e_s mean -gtg_iter 10
#python3 app/main.py -gpus 1 -m llmlp_gtg -ds cifar10 -tr 5 -am corr -am_ts mean none -am_s uncertanity -e_s mean -gtg_iter 10

#python3 app/main.py -gpus 1 -m lstmreg_gtg lstmbc_gtg -ds cifar10 -tr 5 -am corr -am_s uncertanity -am_ts mean none -e_s mean -gtg_iter 10

#python3 app/main.py -gpus 1 -m ll -ds cifar10 -tr 5
#python3 app/main.py -gpus 1 -m llmlp_gtg lsmlps_gtg lstmreg_gtg lstmbc_gtg -ds cifar10 -tr 2 -am corr -am_ts mean none -e_s mean -gtg_iter 10 -bsgtgo 128 #--temp_unlab_pool
python3 app/main.py -gpus 1 -m ll_gtg llmlp_gtg lsmlps_gtg lstmreg_gtg lstmbc_gtg -ds cifar10 -tr 1 -am corr -am_ts none mean -e_s mean --temp_unlab_pool #-gtg_iter 30 #-bsgtgo 128 #-am_s uncertanity
