# GTGAL
## Active Learning Strategy using Graph Trusduction Algorithm

Nowadays neural networks are one of the main pillars of artificial intelligence. However, they are data hungry, which means that to get good results on such a problem like image classification or object detection requires a large amount of information. This results in increased time and training costs. Active Learning wants to overcome this problem by finding the most important observations and features that summarize the data set. Having found those we can get results very close to the one obtained by analysing the entire data set. Our work wants to explore this field by applying the Graph Trasduction Game which formulates the classification task as an evolutionary non-cooperative game between N players (samples) with M strategies (labels). Reaching a Nash Equilibria corresponds to find stable point of a dynamical system, one reached, all the samples are labeled consistently. To this end, the selection of samples to be labeled in the AL model, is based on i) tracking the evolution of the entropy along the iteration of the aforementioned dynamical system and ii) creating an ad-hoc payoff function such that similar samples (already seen) are discouraged to emerge in the subsequent iterations. We exploit the performance of our approach on five publicly available image classification benchmark: CIFAR10/100, SVHN, Fashion-MNIST and Tiny-ImageNET.

## Requirements

Change the last row of **requirements.txt** to the path of yout conda enviroments. Then type in the bash the following lines.
```
conda create --name <env> --file requirements.txt
conda activate <env>
```

## Applciation Start Up
```
sbatch run_AL_multi.slurm    # for multi-GPUs training
sbatch run_AL_single.slurm    # for single-GPU training
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
