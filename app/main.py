# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist

from init import get_dataset, get_model
from utils import create_directory, create_ts_dir, \
    plot_loss_curves, plot_res_std_mean, set_seeds, Entropy_Strategy as ES

from strategies.baselines.LearningLoss import LearningLoss
from strategies.baselines.Random import Random
from strategies.baselines.Entropy import Entropy
from strategies.baselines.LearningLoss import LearningLoss
from strategies.competitors.CoreSet import CoreSet
from strategies.GTG import GTG

    
    
from config import cls_config, al_params, det_config
from datetime import datetime
import argparse
import time
from typing import Dict, Any, List, Tuple

import os
import logging
logger = logging.getLogger(__name__)



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasets', type=str, nargs='+', choices=['cifar10', 'cifar100', 'fmnist', 'tinyimagenet', 'voc', 'coco'],
                        required=True, help='Possible datasets to choose')
    parser.add_argument('-i', '--iterations', type=int, nargs=1, required=True, help='Number or iterations of AL benchmark for each dataset')
    parser.add_argument('-s', '--strategy', type=str, nargs=1, choices=['uncertanity', 'diversity', 'mixed'], 
                        required=True, help='Possible query strategy types to choose')
    parser.add_argument('-ts', '--threshold_strategy', type=str, nargs=1, choices=['threshold', 'mean'], 
                        required=True, help='Possible query strategy types to choose')
    parser.add_argument('-t', '--threshold', type=float, nargs=1, required=False, 
                        help='Affinity Matrix Threshold, when threshold_strategy = mean, this is ingnored')
    parser.add_argument('--wandb', action='store_true', 
                        help='Log benchmark stats into Weights & Biases web app service')
                        

    args = parser.parse_args()
    return args



def run_strategies(ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any], gtg_p: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, List[float]]], List[int]]:

    results = { }
    n_lab_obs = [al_p['init_lab_obs'] + (iter * al_p['n_top_k_obs']) for iter in range(al_p['al_iters'])]
    
    methods = [
        Random(ct_p=ct_p, t_p=t_p, al_p=al_p),
        Entropy(ct_p=ct_p, t_p=t_p, al_p=al_p),
        CoreSet(ct_p=ct_p, t_p=t_p, al_p=al_p),
        LearningLoss(ct_p=ct_p, t_p=t_p, al_p=al_p, LL=True),
        
        GTG(ct_p=ct_p, t_p=t_p, al_p=al_p, 
            gtg_p={**gtg_p, 'rbf_aff': False, 'A_function': 'corr', 'ent_strategy': ES.MEAN}, LL=True), 
        GTG(ct_p=ct_p, t_p=t_p, al_p=al_p, 
            gtg_p={**gtg_p, 'rbf_aff': False, 'A_function': 'corr', 'ent_strategy': ES.H_INT}, LL=True),
    ]
    
    for method in methods:
        logger.info(f'-------------------------- {method.method_name} --------------------------\n')
        results[method.method_name] = method.run()
    return results, n_lab_obs



def main() -> None:
    
    args = get_args()
    choosen_datasets = args.datasets
    wandb = args.wandb
    trials = args.iterations[0]
    strategy_type = args.strategy[0]
    treshold_strategy = args.threshold_strategy[0]
    treshold = args.threshold[0]

    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    create_directory(f'results/{timestamp}')
    
    logging.basicConfig(filename=f'results/{timestamp}/AL_DDP.log',
                    filemode='a',
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")
        device = torch.device('cuda')
        if dist.is_available(): logger.info('We can train on multiple GPU')
        if dist.is_nccl_available(): logger.info('NCCL backend available')
    else:
        logger.info("CUDA is not available. Using CPU.")
        device = torch.device('cpu')

    logger.info(f'Application running on {device}\n')
    
    # setting seed and deterministic behaviour of pytorch for reproducibility
    set_seeds()
            
        
    gtg_params = {
        'gtg_tol': 0.001,
        'gtg_max_iter': 30,
        'strategy_type': strategy_type,
        'threshold_strategy': treshold_strategy,
        'threshold': treshold
    }
    
        
    for dataset_name in choosen_datasets:
        
        if dataset_name in ['cifar10', 'cifar100', 'fmnist', 'tinyimagenet']: task = 'clf'
        else: task = 'detection'

        logger.info(f'----------------------- RUNNING ACTIVE LEARNING BENCHMARK ON {dataset_name} -----------------------\n')

        for trial in range(trials):
            
            logger.info(f'----------------------- SAMPLE ITERATION {trial + 1} / {trials} -----------------------\n')
            
            create_ts_dir(timestamp, dataset_name, str(trial))
            
            Dataset = get_dataset(task, dataset_name, init_lab_obs = al_params['init_lab_obs'])
            Model = get_model(Dataset.image_size, Dataset.n_classes, Dataset.n_channels, device, task)
            
            logger.info('\n')
            
            common_training_params = {
                'Dataset': Dataset, 'Model': Model,
                'device': device, 'timestamp': timestamp,
                'dataset_name': dataset_name,
                'trial': trial, 'task': task,
                'wandb_logs': wandb
            }

            task_params = cls_config if task == 'clf' else det_config

            results, n_lab_obs = run_strategies(
                ct_p = common_training_params, 
                t_p = task_params,
                al_p = al_params,
                gtg_p = gtg_params,
            )
            
            plot_loss_curves(results, n_lab_obs, timestamp,
                             list(task_params['results_dict']['test'].keys()),
                             plot_png_name=f'{dataset_name}/{trial}/results.png')
            
        plot_res_std_mean(task, timestamp, dataset_name)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "ppv-gpu1"
    os.environ["MASTER_PORT"] = "16217"
    
    start = time.time()
    main()
    end = time.time()
    
    logger.info(f'####################################################### TOTAL ELAPSED SECONDS: {end-start} #######################################################')