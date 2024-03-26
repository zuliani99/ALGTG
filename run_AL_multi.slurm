#!/bin/bash
#SBATCH --job-name=AL_multi    # Job name
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=875532@stud.unive.it     # Where to send mail	
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=empty_log.log   # Standard output and error log
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2    # number of GPUs per node
#SBATCH --gpus-per-task=1      # number of GPUs per process


export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
#export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

conda activate al_env

python3 app/main.py -d cifar10 -s mixed -i 1 #2