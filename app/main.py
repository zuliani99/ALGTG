# -*- coding: utf-8 -*-

import torch
torch.autograd.set_detect_anomaly(True) # type: ignore
import torch.distributed as dist

from models.BBone_Module import Master_Model
from models.backbones.ResNet18 import ResNet
from models.backbones.ssd_pytorch.SSD import SSD
from init import get_backbone, get_dataset, get_ll_module_params, get_module
from utils import create_directory, create_ts_dir, plot_loss_curves, \
    plot_res_std_mean, set_seeds, Entropy_Strategy as ES

from strategies.baselines.Random import Random
from strategies.baselines.Entropy import Entropy
from strategies.baselines.LearningLoss import LearningLoss
from strategies.competitors.CoreSet import CoreSet
from strategies.competitors.BADGE import BADGE
from strategies.competitors.BALD import BALD
from strategies.competitors.CDAL import CDAL
from strategies.competitors.K_Means import K_Means
from strategies.competitors.LeastConfidence import LeastConfidence
from strategies.GTG import GTG
from strategies.GTG_LL import GTG_LL
    
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
    parser.add_argument('-m', '--methods', type=str, nargs='+', choices=[
            'random', 'entropy', 'coreset', 'badge', 'bald', 'cdal', 'kmeans', 'leastconfidence',
            'll', 'gtg_ll', 'lq_gtg'
        ],
        required=True, help='Possible methods to choose')
    parser.add_argument('-ds', '--datasets', type=str, nargs='+', choices=['cifar10', 'cifar100', 'fmnist', 'tinyimagenet', 'voc', 'coco'],
                        required=True, help='Possible datasets to choose')
    parser.add_argument('-tr', '--trials', type=int, nargs=1, required=True, help='Number or trials of AL benchmark for each dataset')
    parser.add_argument('-s', '--strategy', type=str, nargs=1, choices=['uncertanity', 'diversity', 'mixed'], 
                        required=True, help='Possible query strategy types to choose for our method')
    parser.add_argument('-ts', '--threshold_strategy', type=str, nargs=1, choices=['threshold', 'mean'], 
                        required=False, help='Possible query strategy types to choose for our method')
    parser.add_argument('-t', '--threshold', type=float, nargs=1, required=False, 
                        help='Affinity Matrix Threshold for our method, when threshold_strategy = mean, this is ingnored')
    parser.add_argument('-plb', '--perc_labeled_batch', type=float, nargs=1, required=False, 
                        help='Number of labeled observations to mantain in each batch during out method')
    parser.add_argument('--wandb', action='store_true', 
                        help='Log benchmark stats into Weights & Biases web app service')
                        

    args = parser.parse_args()
    return args


dict_strategies = dict(
    random = Random, entropy = Entropy, coreset = CoreSet, bald = BALD, badge = BADGE, # -> BB
    kmeans = K_Means, leastconfidence = LeastConfidence, cdal = CDAL, # -> BB
    
    ll = LearningLoss, gtg_ll = GTG_LL, # -> BB + LL
    lq_gtg = GTG # -> BB + GTG
)


def get_strategies_object(methods: List[str], list_gtg_p: List[Dict[str, Any]], Masters: Dict[str, Master_Model], 
                          ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any]) -> List[Any]:
    strategies = []
    for method in methods:
        gtg = method.split('_')
        if 'gtg' in gtg:
            if gtg[0] == 'gtg':
                # test all the gtg configurations
                for gtg_p in list_gtg_p:
                    strategies.append(dict_strategies[method]({**ct_p, 'Master_Model': Masters['M_LL']}, t_p, al_p, gtg_p))
            else:
                # test all the gtg configurations
                for gtg_p in list_gtg_p:
                    strategies.append(dict_strategies[method]({**ct_p, 'Master_Model': Masters['M_GTG']}, t_p, al_p, gtg_p))
        elif method == 'll':
            strategies.append(dict_strategies[method]({**ct_p, 'Master_Model': Masters['M_LL']}, t_p, al_p))
        else:
            strategies.append(dict_strategies[method]({**ct_p, 'Master_Model': Masters['M_None']}, t_p, al_p))
    
    return strategies
    


def run_strategies(ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any], gtg_p: Dict[str, Any],
                   Masters: Dict[str, Master_Model], methods: List[str]) -> Tuple[Dict[str, Dict[str, List[float]]], List[int]]:

    results = { }
    n_lab_obs = [al_p['init_lab_obs'] + (iter * al_p['n_top_k_obs']) for iter in range(al_p['al_iters'])]
    
    # different gtg configurations that we want to test
    list_gtg_p = [
        {**gtg_p, 'rbf_aff': False, 'A_function': 'corr', 'ent_strategy': ES.MEAN},
        {**gtg_p, 'rbf_aff': False, 'A_function': 'corr', 'ent_strategy': ES.H_INT}
    ]
    
    # get the strategis object to run them
    strategies = get_strategies_object(methods, list_gtg_p, Masters, ct_p, t_p, al_p)
    
    for strategy in strategies:
        logger.info(f'-------------------------- {strategy.strategy_name} --------------------------\n')
        results[strategy.strategy_name] = strategy.run()
    
    return results, n_lab_obs


# to create a single master model for each type
def get_masters(methods: List[str], BBone: ResNet | SSD,
                ll_module_params: Dict[str, Any], gtg_module_params: Dict[str, Any],
                dataset_name: str) -> Dict[str, Master_Model]:
    LL_Mod, GTG_Mod, M_None = None, None, None
    ll, only_bb = False, False
    for method in methods:
        method_module = method.split('_')[-1]
        if method_module == 'll' and not ll:
            LL_Mod = get_module('LL', ll_module_params)
            ll = True
        elif method_module == 'gtg':
            GTG_Mod = get_module('GTG', gtg_module_params)
        elif not only_bb:
            M_None = Master_Model(BBone, None, dataset_name)
            only_bb = True
            
    Masters = { }
    if M_None != None: Masters['M_None'] = M_None
    if GTG_Mod != None: Masters['M_GTG'] = Master_Model(BBone, GTG_Mod, dataset_name)
    if LL_Mod != None: Masters['M_LL'] = Master_Model(BBone, LL_Mod, dataset_name)
    
    return Masters                  



def main() -> None:
    
    args = get_args()
    methods = args.methods
    choosen_datasets = args.datasets
    wandb = args.wandb
    perc_labeled_batch = args.perc_labeled_batch[0]
    trials = args.trials[0]
    strategy_type = args.strategy[0]
    treshold_strategy = args.threshold_strategy[0]
    treshold = args.threshold[0]

    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    create_directory(f'results/{timestamp}')
    
    logging.basicConfig(filename = f'results/{timestamp}/AL_DDP.log',
                    filemode = 'a',
                    format = '%(asctime)s - %(levelname)s: %(message)s',
                    datefmt = '%H:%M:%S',
                    level = logging.INFO)
    
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
        'threshold': treshold,
        'perc_labeled_batch': perc_labeled_batch
    }
    
        
    for dataset_name in choosen_datasets:
        
        if dataset_name in ['cifar10', 'cifar100', 'fmnist', 'tinyimagenet']: task = 'clf'
        else: task = 'detection'
        
        task_params = cls_config if task == 'clf' else det_config

        logger.info(f'----------------------- RUNNING ACTIVE LEARNING BENCHMARK ON {dataset_name} -----------------------\n')

        for trial in range(trials):
            
            logger.info(f'----------------------- SAMPLE ITERATION {trial + 1} / {trials} -----------------------\n')
            
            create_ts_dir(timestamp, dataset_name, str(trial))
            
            Dataset = get_dataset(task, dataset_name, init_lab_obs = al_params['init_lab_obs']) # get the dataset
            
            BBone = get_backbone(Dataset.image_size, Dataset.n_classes, Dataset.n_channels, task) # the backbone is the same for all
            
            # create gtg dictionary parameters
            gtg_module_params = dict(
                **gtg_params, n_top_k_obs = al_params['n_top_k_obs'], 
                n_classes = Dataset.n_classes, 
                init_lab_obs = al_params['init_lab_obs'], 
                embedding_dim = BBone.get_embedding_dim(),
                device = device, 
            )
            # create learnin loss dictionary parameters
            ll_module_params = get_ll_module_params(task)
            
            
            # obtain the master models
            Masters = get_masters(methods, BBone, ll_module_params, gtg_module_params, dataset_name)
                        
            logger.info('\n')
            
            common_training_params = {
                'Dataset': Dataset, 'device': device, 'timestamp': timestamp,
                'dataset_name': dataset_name, 'trial': trial, 'task': task,
                'wandb_logs': wandb
            }

            results, n_lab_obs = run_strategies(
                ct_p = common_training_params, 
                t_p = task_params,
                al_p = al_params,
                gtg_p = gtg_params,
                Masters = Masters,
                methods = methods
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