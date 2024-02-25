# -*- coding: utf-8 -*-

import torch

from Datasets import SubsetDataloaders

from strategies.GTG import GTG
from strategies.Random import Random
from strategies.Entropy import Entropy

from utils import create_ts_dir_res, accuracy_score, plot_loss_curves

from datetime import datetime
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', type=str, nargs='+', choices=['cifar10', 'cifar100', 'fmnist'], required=True, help='Possible datasets to chosoe')

args = parser.parse_args()

choosen_datasets = args.datasets


save_plot = True
torch.backends.cudnn.deterministic = True


def train_evaluate(al_params, epochs, len_lab_train_ds, al_iters, unlab_sample_dim, n_top_k_obs, our_method_params):

    results = { }
    n_lab_obs = [len_lab_train_ds + (iter * n_top_k_obs) for iter in range(al_iters)]
    
    methods = [
        # random
        #Random(al_params, LL=False), Random(al_params, LL=True),
        
        # entropy
        #Entropy(al_params, LL=False), Entropy(al_params, LL=True),
        
        # gtg cos_sim -> LL=False, cos_sim and corr as get_A
        GTG(al_params, our_method_params, LL=False, A_function='cos_sim'), GTG(al_params, our_method_params, LL=False, A_function='corr'),
        
        # gtg cos_sim -> LL=True, cos_sim and corr as get_A
        GTG(al_params, our_method_params, LL=True, A_function='cos_sim'), GTG(al_params, our_method_params, LL=True, A_function='corr'),
        
        #########################################################
        Random(al_params, LL=False), Random(al_params, LL=True),
        #########################################################
        
    ]
    
    torch.backends.cudnn.benchmark = False
        
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
    al_iters = 10
    n_top_k_obs = 1000
    unlab_sample_dim = 10000
    batch_size = 128
    patience = 50
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    
    for dataset_name in choosen_datasets:
        print(f'----------------------- RUNING ACTIVE LEARNING BENCHMARK ON {dataset_name} -----------------------\n')
        
        create_ts_dir_res(timestamp, dataset_name)
        
        DatasetChoice = SubsetDataloaders(dataset_name, batch_size, val_rateo=0.2, init_lab_obs=1000, al_iters=al_iters)
        
        
        al_params = {
            'DatasetChoice': DatasetChoice,
            'batch_size': batch_size,
            'score_fn': accuracy_score,
            'device': device,
            'patience': patience,
            'timestamp': timestamp,
            'dataset_name': dataset_name
        }    
        
        
        our_method_params = {
            'gtg_tol': 0.001,
            'gtg_max_iter': 30, #20
        }
        
                                                        
        results, n_lab_obs = train_evaluate(
            al_params=al_params, 
            epochs=epochs, 
            len_lab_train_ds=len(DatasetChoice.lab_train_subset),
            al_iters=al_iters, 
            unlab_sample_dim=unlab_sample_dim,
            n_top_k_obs=n_top_k_obs,
            our_method_params=our_method_params
        )
        
        final_plot_name = f'{dataset_name}/results_{epochs}_{al_iters}_{n_top_k_obs}.png'
        
        
        plot_loss_curves(results, n_lab_obs, save_plot, timestamp, final_plot_name)
    

if __name__ == "__main__":
    main()