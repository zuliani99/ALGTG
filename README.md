# GTGAL
## Active Learning Strategy using Graph Trusduction Algorithm

Nowadays neural networks are one of the main pillars of artificial intelligence. However, they are data hungry, which means that to get good results on such a problem like image classification or object detection requires a large amount of information. This results in increased time and training costs. Active Learning wants to overcome this problem by finding the most important observations and features that summarize the data set. Having found those we can get results very close to the one obtained by analysing the entire data set. 

Our work wants to explore this field by applying the **Graph Trasduction Game** which formulates the classification task as an *evolutionary non-cooperative game* between *N players* (samples) with *M strategies* (labels). Reaching a **Nash Equilibria** corresponds to find stable point of a dynamical system, one reached, all the samples are labeled consistently. To this end, the selection of samples to be labeled in the AL model, is based on:

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
               {random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,lq_gtg}
               [{random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,lq_gtg} ...]
               -ds
               {cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet,voc,coco}
               [{cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet,voc,coco} ...]
               [-tr TRIALS]
               [-am {corr,cos_sim,rbfk} [{corr,cos_sim,rbfk} ...]]
               [-am_s {uncertanity,diversity,mixed}]
               [-am_ts {threshold,mean,none} [{threshold,mean,none} ...]]
               [-am_t AFFINITY_MATRIX_THRESHOLD] [-e_s {mean,integral}]
               [-gtg_iter GTG_ITERATIONS] [-gtg_t GTG_TOLLERANCE]
               [-plb PERC_LABELED_BATCH] [--wandb]

options:
  -h, --help            show this help message and exit
  -gpus GPUS, --gpus GPUS
                        Number of GPUs to use during training
  -m {random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,lq_gtg} [{random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,lq_gtg} ...],
     --methods {random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,lq_gtg} [{random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,lq_gtg} ...]
                        Possible methods to choose
  -ds {cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet,voc,coco} [{cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet,voc,coco} ...],
     --datasets {cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet,voc,coco} [{cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet,voc,coco} ...]
                        Possible datasets to choose
  -tr TRIALS, --trials TRIALS
                        AL trials
  -am {corr,cos_sim,rbfk} [{corr,cos_sim,rbfk} ...], --affinity_matrix {corr,cos_sim,rbfk} [{corr,cos_sim,rbfk} ...]
                        Affinity matrix to choose
  -am_s {uncertanity,diversity,mixed}, --affinity_matrix_strategy {uncertanity,diversity,mixed}
                        Different affinity matrix modification
  -am_ts {threshold,mean,none} [{threshold,mean,none} ...], --affinity_matrix_threshold_strategies {threshold,mean,none} [{threshold,mean,none} ...]
                        Possible treshold strategy types to choose to apply in
                        the affinity matrix
  -am_t AFFINITY_MATRIX_THRESHOLD, --affinity_matrix_threshold AFFINITY_MATRIX_THRESHOLD
                        Affinity Matrix Threshold for our method, when
                        threshold_strategy = mean, this is ignored
  -e_s {mean,integral}, --entropy_strategy {mean,integral}
                        Entropy strategy to sum up the entropy history
  -gtg_iter GTG_ITERATIONS, --gtg_iterations GTG_ITERATIONS
                        Maximum GTG iterations to perorm
  -gtg_t GTG_TOLLERANCE, --gtg_tollerance GTG_TOLLERANCE
                        GTG tollerance
  -plb PERC_LABELED_BATCH, --perc_labeled_batch PERC_LABELED_BATCH
                        Number of labeled observations to mantain in each
                        batch during GTG end-to-end version
  --wandb               Log benchmark stats into Weights & Biases web app
                        service
```

## Notes
Github url repo:
```
git clone https://github.com/zuliani99/Active_Learning_GTG.git
```

GitHub Token
```
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
```
