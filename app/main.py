# -*- coding: utf-8 -*-

import torch

from Datasets import SubsetDataloaders

from strategies.baselines.Random import Random
from strategies.baselines.Entropy import Entropy
from strategies.GTG import GTG
from strategies.competitors.CoreSet import CoreSet
from strategies.competitors.BALD import BALD
from strategies.competitors.BADGE import BADGE
from strategies.competitors.LeastConfidence import LeastConfidence
from strategies.competitors.K_Means import K_Means

from utils import create_ts_dir_res, accuracy_score, plot_loss_curves, plot_accuracy_std_mean, set_seeds, Entropy_Strategy

from datetime import datetime
import argparse
import time
from typing import Dict, Any, List, Tuple




parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', type=str, nargs='+', choices=['cifar10', 'cifar100', 'fmnist', 'tinyimagenet'],
                    required=True, help='Possible datasets to choose')
parser.add_argument('-i', '--iterations', type=int, nargs=1, required=True, help='Number or iterations of AL benchmark for each dataset')
parser.add_argument('-s', '--strategy', type=str, nargs=1, choices=['uncertanity', 'diversity', 'mixed'], 
                    required=True, help='Possible query strategy types to choose')


args = parser.parse_args()

choosen_datasets = args.datasets
sample_iterations = args.iterations[0]
strategy_type = args.strategy[0]


# setting seed and deterministic behaviour of pytorch for reproducibility
set_seeds()



def train_evaluate(training_params: Dict[str, Any], gtg_params: Dict[str, int], al_params: Dict[str, Any], \
    epochs: int, len_lab_train_ds: int) -> Tuple[Dict[str, List[float]], List[int]]:

    results = { }
    n_lab_obs = [len_lab_train_ds + (iter * al_params['n_top_k_obs']) for iter in range(al_params['al_iters'])]
    
    methods = [
        # Random
        #Random(al_params=al_params, training_params=training_params, LL=False),
        Random(al_params=al_params, training_params=training_params, LL=True),
        
        # LeastConfidence
        #LeastConfidence(al_params=al_params, training_params=training_params, LL=False)
        #LeastConfidence(al_params=al_params, training_params=training_params, LL=True)
        
        # Rntropy
        #Entropy(al_params=al_params, training_params=training_params, LL=False)
        #Entropy(al_params=al_params, training_params=training_params, LL=True),
        
        # KMeans
        #K_Means(al_params=al_params, training_params=training_params, LL=True),
        #K_Means(al_params=al_params, training_params=training_params, LL=True),
        
        # CoreSet
        #CoreSet(al_params=al_params, training_params=training_params, LL=True),
        #CoreSet(al_params=al_params, training_params=training_params, LL=True),
        
        # BALD
        #BALD(al_params=al_params, training_params=training_params, LL=True),
        #BALD(al_params=al_params, training_params=training_params, LL=True),
        
        # BADGE
        #BADGE(al_params=al_params, training_params=training_params, LL=True), 
        #BADGE(al_params=al_params, training_params=training_params, LL=True),
        
        # GTG
        GTG(al_params=al_params, training_params=training_params, 
            gtg_params={
                **gtg_params,
                'rbf_aff': False, 'A_function': 'cos_sim', 'zero_diag': False, 'ent_strategy': Entropy_Strategy.WEIGHTED_AVERAGE_DERIVATIVES
            }, LL=True),
        
        GTG(al_params=al_params, training_params=training_params,
            gtg_params={
                **gtg_params,
                'rbf_aff': False, 'A_function': 'cos_sim', 'zero_diag': False, 'ent_strategy': Entropy_Strategy.HISTORY_INTEGRAL
            }, LL=True),
        
        GTG(al_params=al_params, training_params=training_params,
            gtg_params={
                **gtg_params,
                'rbf_aff': True, 'A_function': 'corr', 'zero_diag': False, 'ent_strategy': Entropy_Strategy.HISTORY_INTEGRAL
            }, LL=True)
    ]
    '''GTG(al_params=al_params, training_params=training_params, 
            gtg_params={
                **gtg_params,
                'rbf_aff': False, 'A_function': 'corr', 'zero_diag': False, 'ent_strategy': Entropy_Strategy.WEIGHTED_AVERAGE_DERIVATIVES
            }, LL=True),
        GTG(al_params=al_params, training_params=training_params, 
            gtg_params={
                **gtg_params,
                'rbf_aff': False, 'A_function': 'corr', 'zero_diag': False, 'ent_strategy': Entropy_Strategy.HISTORY_INTEGRAL
            }, LL=True),
        
        GTG(al_params=al_params, training_params=training_params,
            gtg_params={
                **gtg_params,
                'rbf_aff': True, 'A_function': 'corr', 'zero_diag': False, 'ent_strategy': Entropy_Strategy.WEIGHTED_AVERAGE_DERIVATIVES
            }, LL=True),
        GTG(al_params=al_params, training_params=training_params,
            gtg_params={
                **gtg_params,
                'rbf_aff': True, 'A_function': 'corr', 'zero_diag': False, 'ent_strategy': Entropy_Strategy.HISTORY_INTEGRAL
            }, LL=True)'''

    for method in methods:
            
        print(f'-------------------------- {method.method_name} --------------------------\n')
            
        results[method.method_name] = method.run(epochs)
            
                    
    print('Resulting dictionary')
    print(f'{results}\n')
        
    print('Resulting number of observations')
    print(f'{n_lab_obs}\n')
        
    return results, n_lab_obs



def main() -> None:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("Number of available GPUs:", num_gpus)
        device = torch.device('cuda')
        if torch.distributed.is_available(): print('We can train on multiple GPU')
        if torch.distributed.is_nccl_available(): print('NCCL backend available')
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device('cpu')

    print(f'Application running on {device}\n')


    # later added to argparser
    al_iters = 2#10
    n_top_k_obs = 1000
    unlab_sample_dim = 10000
    init_lab_obs = 1000
    
    epochs = 10#200
    batch_size = 128
    patience = 50
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
        
    for dataset_name in choosen_datasets:

        print(f'----------------------- RUNNING ACTIVE LEARNING BENCHMARK ON {dataset_name} -----------------------\n')

        for samp_iter in range(sample_iterations):
            
            print(f'----------------------- SAMPLE ITERATION {samp_iter + 1} / {sample_iterations} -----------------------\n')
            
            create_ts_dir_res(timestamp, dataset_name, str(samp_iter))
            DatasetChoice = SubsetDataloaders(dataset_name, batch_size, val_rateo=0.2, init_lab_obs=init_lab_obs)
            
            print('\n')
            
            training_params = {
                'DatasetChoice': DatasetChoice,
                'batch_size': batch_size,
                'score_fn': accuracy_score,
                'device': device,
                'patience': patience,
                'timestamp': timestamp,
                'dataset_name': dataset_name,
                'samp_iter': samp_iter
            }
            
            al_params = {
                'al_iters': al_iters, 
                'unlab_sample_dim': unlab_sample_dim, 
                'n_top_k_obs': n_top_k_obs,
            }
            
            gtg_params = {
                'gtg_tol': 0.001,
                'gtg_max_iter': 30,
                'strategy_type': strategy_type
                # remember that simpson's rule need an even number of observations to compute the integrals
            }
            

            results, n_lab_obs = train_evaluate(
                training_params=training_params, 
                al_params=al_params,
                gtg_params=gtg_params,
                epochs=epochs, 
                len_lab_train_ds=init_lab_obs,
            )
            
            final_plot_name = f'{dataset_name}/{samp_iter}/results_{epochs}_{al_iters}_{n_top_k_obs}.png'
            
            plot_loss_curves(results, n_lab_obs, timestamp, plot_png_name=final_plot_name)
            
        plot_accuracy_std_mean(timestamp, dataset_name)


if __name__ == "__main__":
    
    start = time.time()
    main()
    end = time.time()
    
    print(f'####################################################### TOTAL ELAPSED SECONDS: {end-start} #######################################################')