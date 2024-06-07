# ALGTG
## Active Learning trategy using Graph Trusduction Game

**Active Learning (AL)** aims to leverage the performance of deep models by selecting the most valuable samples to be annotated from a pool of unlabelled data to be annotated and moved into the labeled-training set. On the other hand **Game Theory** deals with the study of strategic interactions between rational decision-makers, which are modeled as games.

Our work wants to connects these two fields by applying the **Graph Trasduction Game** which formulates the classification task as an *evolutionary non-cooperative game* between *N players* (samples) with *M strategies* (labels). Reaching a **Nash Equilibria** corresponds to find stable point of a dynamical system, one reached, all the samples are labeled consistently. To this end, the selection of samples to be labelled in the AL model, is based on:

1. Tracking the evolution of the **entropy** along the iteration of the aforementioned dynamical system
2. Creating an ad-hoc payoff function such that similar samples (already seen) are discouraged to emerge in the subsequent iterations.

We exploit the performance of our approach on five publicly available image classification benchmark: **CIFAR10/100**, **SVHN**, **Fashion-MNIST** and **Tiny-ImageNET**.

## Requirements

Change the last row of **requirements.txt** to the path of yout conda enviroments. Then type in the bash the following lines.
```
conda create --name <env> --file requirements.txt
conda activate <env>
```

## Applciation Start Up
Example of application start-up:
```
python3 app/main.py -gpus 1 -m random entropy coreset badge bald cdal tavaal ll ll_gtg -ds cifar10 -tr 5 -am corr -am_s mixed -am_ts mean -e_s mean
```

Accepted Arguments:
```
usage: main.py [-h] [-gpus GPUS] -m
               {random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg}
               [{random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg} ...]
               -ds {cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet}
               [{cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet} ...]
               [-tr TRIALS]
               [-am {corr,cos_sim,rbfk} [{corr,cos_sim,rbfk} ...]]
               [-am_s {uncertanity,diversity,mixed}]
               [-am_ts {threshold,mean,none} [{threshold,mean,none} ...]]
               [-am_t AFFINITY_MATRIX_THRESHOLD] [-e_s {mean,integral}]
               [-gtg_iter GTG_ITERATIONS] [-gtg_t GTG_TOLLERANCE]
               [-plb PERC_LABELLED_BATCH]

options:
  -h, --help            show this help message and exit
  -gpus GPUS, --gpus GPUS
                        Number of GPUs to use during training
  -m {random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg}
      [{random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg} ...],
      --methods {random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg} [{random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg} ...]
                        Possible methods to choose
  
  -ds {cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet} [{cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet} ...], 
      --datasets {cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet} [{cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet} ...]
                        Possible datasets to choose
  
  -tr TRIALS, --trials TRIALS
                        AL trials
  
  -am {corr,cos_sim,rbfk} [{corr,cos_sim,rbfk} ...], --affinity_matrix {corr,cos_sim,rbfk} [{corr,cos_sim,rbfk} ...]
                        Affinity matrix to choose
  
  -am_s {uncertanity,diversity,mixed}, --affinity_matrix_strategy {uncertanity,diversity,mixed}
                        Different affinity matrix modification
  
  -am_ts {threshold,mean,none} [{threshold,mean,none} ...], --affinity_matrix_threshold_strategies {threshold,mean,none} [{threshold,mean,none} ...]
                        Possible treshold strategy types to choose to apply in the affinity matrix
  
  -am_t AFFINITY_MATRIX_THRESHOLD, --affinity_matrix_threshold AFFINITY_MATRIX_THRESHOLD
                        Affinity Matrix Threshold for our method, when threshold_strategy = mean, this is ignored
  
  -e_s {mean,integral}, --entropy_strategy {mean,integral}
                        Entropy strategy to sum up the entropy history
  
  -gtg_iter GTG_ITERATIONS, --gtg_iterations GTG_ITERATIONS
                        Maximum GTG iterations to perorm
  
  -gtg_t GTG_TOLLERANCE, --gtg_tollerance GTG_TOLLERANCE
                        GTG tollerance
  
  -plb PERC_LABELLED_BATCH, --perc_labelled_batch PERC_LABELLED_BATCH
                        Number of labelled observations to mantain in each batch during GTG end-to-end version
```

