# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist

from .datasets_creation.Classification import Cls_Datasets
from .datasets_creation.Detection import Det_Datasets
from .models import get_resnet_model, get_ssd_model
from .strategies.baselines.Random import Random
from .strategies.baselines.Entropy import Entropy
from .strategies.GTG import GTG

from .utils import create_directory, create_ts_dir, plot_loss_curves, plot_accuracy_std_mean, set_seeds, Entropy_Strategy
    
    
from .config import cls_config, al_params, det_config
from datetime import datetime
import argparse
import time
from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)




def get_dataset(task, dataset_name, init_lab_obs):
    if task == 'clf': return Cls_Datasets(dataset_name, init_lab_obs=init_lab_obs)
    else: return Det_Datasets(dataset_name, init_lab_obs=init_lab_obs)


def get_model(n_classes, device, task):
    if task == 'clf': return get_resnet_model(n_classes, device)
    else: return get_ssd_model(n_classes, device)




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasets', type=str, nargs='+', choices=['cifar10', 'cifar100', 'fmnist', 'tinyimagenet', 'voc', 'coco'],
                        required=True, help='Possible datasets to choose')
    parser.add_argument('-i', '--iterations', type=int, nargs=1, required=True, help='Number or iterations of AL benchmark for each dataset')
    parser.add_argument('-s', '--strategy', type=str, nargs=1, choices=['uncertanity', 'diversity', 'mixed'], 
                        required=True, help='Possible query strategy types to choose')
    args = parser.parse_args()
    return args



def run_strategies(common_training_params: Dict[str, Any], task_params: Dict[str, Any], al_params: Dict[str, Any], gtg_params: Dict[str, Any],\
    len_lab_train_ds: int) -> Tuple[Dict[str, List[float]], List[int]]:

    results = { }
    n_lab_obs = [len_lab_train_ds + (iter * al_params['n_top_k_obs']) for iter in range(al_params['al_iters'])]
    
    strategy_dict_params = dict(
        common_training_params=common_training_params,
        task_params=task_params, al_params=al_params, 
        LL=True
    )
    
    
    methods = [
        
        Random(strategy_dict_params),
        
    ]
    
    for method in methods:
        logger.info(f'-------------------------- {method.method_name} --------------------------\n')
        results[method.method_name] = method.run()
    return results, n_lab_obs



def main() -> None:
    
    args = get_args()
    choosen_datasets = args.datasets
    trials = args.iterations[0]
    strategy_type = args.strategy[0]

    
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

    if choosen_datasets in ['cifar10', 'cifar100', 'fmnist', 'tinyimagenet']: task = 'clf'
    else: task = 'detection'
    
    init_lab_obs = 1000
    
    
    gtg_params = {
        'gtg_tol': 0.001,
        'gtg_max_iter': 30, # remember that simpson's rule need an even number of observations to compute the integrals
        'strategy_type': strategy_type,
        #'threshold_strategy': treshold_strategy,
        #'threshold': treshold
    }
    
        
    for dataset_name in choosen_datasets:

        logger.info(f'----------------------- RUNNING ACTIVE LEARNING BENCHMARK ON {dataset_name} -----------------------\n')

        for trial in range(trials):
            
            logger.info(f'----------------------- SAMPLE ITERATION {trial + 1} / {trials} -----------------------\n')
            
            create_ts_dir(timestamp, dataset_name, str(trial))
            
            Dataset = get_dataset(task, dataset_name, init_lab_obs=init_lab_obs)
            Model = get_model(dataset_name, device, task)
            
            logger.info('\n')
            
            common_training_params = {
                'Dataset': Dataset,
                'Model': Model,
                'device': device,
                'timestamp': timestamp,
                'dataset_name': dataset_name,
                'trial': trial,
                'task': task
            }

            results, n_lab_obs = run_strategies(
                common_training_params=common_training_params, 
                task_params=cls_config if task == 'clf' else det_config,
                al_params=al_params,
                gtg_params=gtg_params,
                len_lab_train_ds=init_lab_obs,
            )
                        
            plot_loss_curves(results, n_lab_obs, timestamp, plot_png_name=f'{dataset_name}/{trial}/results.png')
            
        plot_accuracy_std_mean(timestamp, dataset_name)


if __name__ == "__main__":
    
    start = time.time()
    main()
    end = time.time()
    
    logger.info(f'####################################################### TOTAL ELAPSED SECONDS: {end-start} #######################################################')