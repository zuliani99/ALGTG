# -*- coding: utf-8 -*-

import torch

torch.autograd.set_detect_anomaly(True) # type: ignore
import torch.distributed as dist

from models.BBone_Module import Master_Model

from init import get_backbone, get_dataset, get_ll_module_params, get_masters, get_strategies_object, dict_backbone
from utils import create_directory, create_ts_dir, plot_trail_acc, plot_res_std_mean, set_seeds, save_yamal

    	
from config import cls_config, al_params, det_config
from datetime import datetime
import argparse
import time
from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpus', '--gpus', type=int, required=False, default=1, help='Number of GPUs to use during training')
    parser.add_argument('-m', '--methods', type=str, nargs='+', required=True, choices=[
            'random', 'entropy', 'coreset', 'badge', 'bald', 'cdal', 'tavaal', 'alphamix', 'tidal',
            'll', 'gtg', 'll_gtg', 'llmlp_gtg', 'lsmlps_gtg', 'lstmreg_gtg', 'lstmbc_gtg'
        ], help='Possible methods to choose')
    parser.add_argument('-ds', '--datasets', type=str, nargs='+', required=True, choices=['cifar10', 'cifar100', 'svhn', 'fmnist', 'caltech256', 'tinyimagenet'], #'voc', 'coco'
                        help='Possible datasets to choose')
    parser.add_argument('-tr', '--trials', type=int, required=False, default=5, help='AL trials')
   
    parser.add_argument('-tulp', '--temp_unlab_pool', required=False, action='store_true', help='Temporary Unlabelled Pool')
    parser.add_argument('-am', '--affinity_matrix', type=str, nargs='+', required=False, choices=['corr', 'cos_sim', 'rbfk'], default=['corr'], help='Affinity matrix to choose')
    
    parser.add_argument('-am_s', '--affinity_matrix_strategy', type=str, required=False, choices=['uncertanity', 'diversity', 'mixed'], default='mixed', 
                       help='Different affinity matrix modification')
    parser.add_argument('-am_ts', '--affinity_matrix_threshold_strategies', type=str, required=False, nargs='+', choices=['threshold', 'mean', 'none'], default=['mean', 'none'],
                        help='Possible treshold strategy types to choose to apply in the affinity matrix')
    parser.add_argument('-am_t', '--affinity_matrix_threshold', type=float, required=False, default=0.5, 
                        help='Affinity Matrix Threshold for our method, when threshold_strategy = mean, this is ignored')
    parser.add_argument('-e_s', '--entropy_strategy', type=str, required=False, choices=['mean', 'integral'], default='mean',
                        help='Entropy strategy to sum up the entropy history')
    
    parser.add_argument('-gtg_iter', '--gtg_iterations', type=int, required=False, default=30, help='Maximum GTG iterations to perorm')
    parser.add_argument('-gtg_t', '--gtg_tollerance', type=float, required=False, default=0.0001, help='GTG tollerance')
        
    #parser.add_argument('-plb', '--perc_labelled_batch', type=float,  required=False, default=0.5,
    #                    help='Number of labelled observations to mantain in each batch during GTG end-to-end version')
    parser.add_argument('-bsgtgo', '--batch_size_gtg_online', type=int,  required=False, default=32, help='Initial batch size for the online GTG version')

    args = parser.parse_args()
    return args




def run_strategies(ct_p: Dict[str, Any], t_p: Dict[str, Any], gtg_p: Dict[str, Any],
                   Masters: Dict[str, Master_Model], methods: List[str]) -> Tuple[Dict[str, Dict[str, List[float]]], List[int]]:
    #Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[int, Dict[str, int]]], List[int]]:

    results = { }
    #count_classes = { }
    n_lab_obs = [al_params["init_lab_obs"] + (iter * al_params["n_top_k_obs"]) for iter in range(al_params["al_iters"])]
     
    # get the strategis object to run them
    strategies = get_strategies_object(methods, Masters, ct_p, t_p, gtg_p)
    
    ##########################################################################################################
    # START THE ACTIVE LEARNING PROECSS
    for strategy in strategies:
        logger.info(f'-------------------------- {strategy.strategy_name} --------------------------\n')
        #results[strategy.strategy_name], count_classes[strategy.strategy_name] = strategy.run()
        results[strategy.strategy_name] = strategy.run()
    ##########################################################################################################
    
    #return results, count_classes, n_lab_obs               
    return results, n_lab_obs               


def get_device(args) -> torch.device:
    if torch.cuda.is_available():
        logger.info(f'Using {args.gpus} / {torch.cuda.device_count()} of the available GPUs')
        device = torch.device('cuda')
        if dist.is_available(): logger.info('We can train on multiple GPU')
        if dist.is_nccl_available(): logger.info('NCCL backend available')
    else:
        logger.info('CUDA is not available. Using CPU')
        device = torch.device('cpu')

    logger.info(f'Application running on {device}\n')
    
    return device



def main() -> None:
    args = get_args()
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    create_directory(f'results/{timestamp}')
    
    logging.basicConfig(filename = f'results/{timestamp}/AL_DDP.log',
                    filemode = 'a',
                    format = '%(asctime)s - %(levelname)s: %(message)s',
                    datefmt = '%H:%M:%S',
                    level = logging.INFO)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    device = get_device(args)
    
    # setting seed and deterministic behaviour of pytorch for reproducibility
    set_seeds()
            
        
    gtg_params = {
        'gtg_t': args.gtg_tollerance,
        'gtg_i': args.gtg_iterations,
        
        'am': args.affinity_matrix,
        'am_s': args.affinity_matrix_strategy,
        'am_ts': args.affinity_matrix_threshold_strategies,
        'am_t': args.affinity_matrix_threshold,
        'e_s': args.entropy_strategy,
        
        #'plb': args.perc_labelled_batch,
        'bsgtgo': args.batch_size_gtg_online,
    }
    
        
    for dataset_name in args.datasets:
        
        if dataset_name in ['cifar10', 'cifar100', 'svhn', 'caltech256', 'tinyimagenet', 'fmnist']: 
            task, task_params = 'clf', cls_config
        else: task, task_params = 'detection', det_config

        logger.info(f'----------------------- RUNNING ACTIVE LEARNING BENCHMARK ON {dataset_name} -----------------------\n')
        
        Dataset = get_dataset(task, dataset_name) # get the dataset
        BBone = get_backbone(Dataset.n_classes, Dataset.n_channels, dict_backbone[dataset_name]) # the backbone is the same for all

        # create gtg dictionary parameters
        gtg_module_params = dict(
            **gtg_params, n_classes = Dataset.n_classes, 
            embedding_dim = BBone.get_rich_features_shape(),
            device = device
        )
        
        ll_module_params = get_ll_module_params(task, Dataset.image_size, dataset_name) # create learning loss dictionary parameters
            
        Masters = get_masters(args.methods, BBone, ll_module_params, gtg_module_params, dataset_name, Dataset.n_classes) # obtain the master models
                                    
        logger.info(f'args.temp_unlab_pool {args.temp_unlab_pool}')
            
        common_training_params = {
            'Dataset': Dataset, 'device': device, 'timestamp': timestamp,
            'dataset_name': dataset_name, 'task': task, 'gpus': args.gpus,
            'temp_unlab_pool': args.temp_unlab_pool,
            
        }
        
        for trial in range(args.trials):
            common_training_params["trial"] = trial
            
            logger.info(f'----------------------- SAMPLE ITERATION {trial + 1} / {args.trials} -----------------------\n')
            
            create_ts_dir(timestamp, dataset_name, str(trial))
            
            Dataset.get_initial_subsets(trial) # get the random split of dataset
            
            #results, count_classes, n_lab_obs = run_strategies(
            results, n_lab_obs = run_strategies(
                ct_p = common_training_params, t_p = task_params, gtg_p = gtg_params, Masters = Masters, methods = args.methods
            )
            
            plot_trail_acc(dataset_name, trial, results, n_lab_obs, timestamp,
                             list(task_params["results_dict"]["test"].keys())[0],
                             plot_png_name=f'{dataset_name}/{trial}/results.png')
        
        plot_res_std_mean(task, timestamp, dataset_name)

        # saving yamal configuration file
        save_yamal(common_training_params, task_params, al_params, gtg_params, args, timestamp, dataset_name)
            

if __name__ == "__main__":
    
    start = time.time()
    main()
    end = time.time()
    
    logger.info(f'####################################################### TOTAL ELAPSED SECONDS: {end-start} #######################################################')
