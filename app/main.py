# -*- coding: utf-8 -*-

import torch

from Datasets import SubsetDataloaders

from strategies.Random import Random
from strategies.Entropy import Entropy
from strategies.GTG import GTG
from strategies.CoreSet import CoreSet
from strategies.BALD import BALD
from strategies.BADGE import BADGE

from utils import create_ts_dir_res, accuracy_score, plot_loss_curves, plot_accuracy_std_mean

from datetime import datetime
import argparse

import numpy as np

# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', type=str, nargs='+', choices=['cifar10', 'cifar100', 'fmnist'],
                    required=True, help='Possible datasets to chosoe')
parser.add_argument('-i', '--iterations', type=int, nargs=1, required=True, help='Number or iterations of AL benchmark for each dataset')

args = parser.parse_args()

choosen_datasets = args.datasets
iterations = args.iterations[0]

# setting seed and deterministic behaviour of pytorch for reproducibility
# https://discuss.pytorch.org/t/determinism-in-pytorch-across-multiple-files/156269
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def train_evaluate(al_params, epochs, len_lab_train_ds, al_iters, unlab_sample_dim, n_top_k_obs, our_method_params):

    results = { }
    n_lab_obs = [len_lab_train_ds + (iter * n_top_k_obs) for iter in range(al_iters)]
    
    methods = [
        # random
        Random(al_params, LL=False), Random(al_params, LL=True),        
        
        # entropy
        Entropy(al_params, LL=False), Entropy(al_params, LL=True),
        
        # coreset
        CoreSet(al_params, LL=False), CoreSet(al_params, LL=True),
        
        # BALD
        BALD(al_params, LL=False), BALD(al_params, LL=True),
        
        # BADGE
        BADGE(al_params, LL=False), BADGE(al_params, LL=True),
        
        # GTG
        #zero_diag=False -> diagonal set to 1       
        GTG(al_params, our_method_params, LL=False, A_function='cos_sim', zero_diag=False),
        GTG(al_params, our_method_params, LL=True, A_function='cos_sim', zero_diag=False),
    ]
        
    for method in methods:
            
        print(f'-------------------------- {method.method_name} --------------------------\n')
            
        results[method.method_name] = method.run(al_iters, epochs, unlab_sample_dim, n_top_k_obs)
            
                    
    print('Resulting dictionary')
    print(results)
    print('\n')
        
    print('Resulting number of observations')
    print(n_lab_obs)
    print('\n')
        
    return results, n_lab_obs



def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Application running on {device}\n')

    epochs = 200
    al_iters = 6
    n_top_k_obs = 1000
    unlab_sample_dim = 10000
    batch_size = 128
    patience = 50
    init_lab_obs = 1000
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
        
    for dataset_name in choosen_datasets:

        print(f'----------------------- RUNNING ACTIVE LEARNING BENCHMARK ON {dataset_name} -----------------------\n')

        for iter in range(iterations):
            
            print(f'----------------------- SAMPLE ITERATION {iter + 1} / {iterations} -----------------------\n')
            
            create_ts_dir_res(timestamp, dataset_name, str(iter))
            
            DatasetChoice = SubsetDataloaders(dataset_name, batch_size, val_rateo=0.2, init_lab_obs=init_lab_obs, al_iters=al_iters)
            
            
            al_params = {
                'DatasetChoice': DatasetChoice,
                'batch_size': batch_size,
                'score_fn': accuracy_score,
                'device': device,
                'patience': patience,
                'timestamp': timestamp,
                'dataset_name': dataset_name,
                'iter': iter
            }    
            
            
            our_method_params = {
                'gtg_tol': 0.001,
                'gtg_max_iter': 30, #20
            }
            
                                                            
            results, n_lab_obs = train_evaluate(
                al_params=al_params, 
                epochs=epochs, 
                len_lab_train_ds=init_lab_obs,
                al_iters=al_iters, 
                unlab_sample_dim=unlab_sample_dim,
                n_top_k_obs=n_top_k_obs,
                our_method_params=our_method_params
            )
            
            final_plot_name = f'{dataset_name}/{iter}/results_{epochs}_{al_iters}_{n_top_k_obs}.png'
            
            
            plot_loss_curves(results, n_lab_obs, timestamp, plot_png_name=final_plot_name)
            
        plot_accuracy_std_mean(timestamp, dataset_name)

if __name__ == "__main__":
    main()