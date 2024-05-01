# -*- coding: utf-8 -*-

import torch

torch.autograd.set_detect_anomaly(True) # type: ignore
import torch.distributed as dist

from models.BBone_Module import Master_Model
from models.backbones.ResNet18 import ResNet
from models.backbones.ssd_pytorch.SSD import SSD
from models.backbones.VGG import VGG

from init import get_backbone, get_dataset, get_ll_module_params, get_module
from utils import create_directory, create_ts_dir, plot_loss_curves, \
    plot_res_std_mean, set_seeds

from strategies.baselines.Random import Random
from strategies.baselines.Entropy import Entropy
from strategies.baselines.LearningLoss import LearningLoss
from strategies.competitors.CoreSet import CoreSet
from strategies.competitors.BADGE import BADGE
from strategies.competitors.CDAL import CDAL
from strategies.competitors.TA_VAAL.TA_VAAL import TA_VAAL
from strategies.GTG import GTG
from strategies.GTG_off import GTG_off
    
from config import cls_config, al_params, det_config
from datetime import datetime
import argparse
import time
from typing import Dict, Any, List, Tuple
import copy

import logging
logger = logging.getLogger(__name__)



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--methods', type=str, nargs='+', required=True, choices=[
            'random', 'entropy', 'coreset', 'badge', 'cdal', 'tavaal',
            'll', 'gtg', 'll_gtg', 'lq_gtg'
        ],help='Possible methods to choose')
    parser.add_argument('-ds', '--datasets', type=str, nargs='+', required=True, choices=['cifar10', 'cifar100', 'svhn', 'fmnist', 'caltech256', 'tinyimagenet', 'voc', 'coco'],
                        help='Possible datasets to choose')
    parser.add_argument('-tr', '--trials', type=int, required=False, default=5, help='AL trials')
    
    parser.add_argument('-am', '--affinity_matrix', type=str, nargs='+', required=False, choices=['corr', 'cos_sim', 'e_d'], default=['corr', 'cos_sim', 'rbfk'],
                        help='Affinity matrix to choose')
    parser.add_argument('-am_s', '--affinity_matrix_strategy', type=str, required=False, choices=['uncertanity', 'diversity', 'mixed'], default='mixed', 
                       help='Different affinity matrix modification')
    parser.add_argument('-am_ts', '--affinity_matrix_threshold_strategy', type=str, required=False, choices=['threshold', 'mean'], default='mean',
                        help='Possible treshold strategy types to choose to apply in the affinity matrix')
    parser.add_argument('-am_t', '--affinity_matrix_threshold', type=float, required=False, default=0.5, 
                        help='Affinity Matrix Threshold for our method, when threshold_strategy = mean, this is ignored')
    parser.add_argument('-e_s', '--entropy_strategy', type=str, required=False, choices=['mean', 'integral'], default='mean',
                        help='Entropy strategy to sum up the entropy history')
    
    parser.add_argument('-gtg_iter', '--gtg_iterations', type=int, required=False, default=30,
                        help='Maximum GTG iterations to perorm')
    parser.add_argument('-gtg_t', '--gtg_tollerance', type=float, required=False, default=0.0001,
                        help='GTG tollerance')
        
    parser.add_argument('-plb', '--perc_labeled_batch', type=float,  required=False, default=0.5,
                        help='Number of labeled observations to mantain in each batch during GTG end-to-end version')
    parser.add_argument('--wandb', action='store_true', 
                        help='Log benchmark stats into Weights & Biases web app service')

    args = parser.parse_args()
    return args


dict_strategies = dict(
    random = Random, entropy = Entropy, coreset = CoreSet, badge = BADGE, # -> BB
    cdal = CDAL, gtg = GTG_off,# -> BB
    
    ll = LearningLoss, ll_gtg = GTG_off, tavaal = TA_VAAL, # -> BB + LL
    lq_gtg = GTG # -> BB + GTG
)

dict_backbone = dict(
    cifar10 = 'ResNet', cifar100 = 'ResNet', svhn = 'ResNet', fmnist = 'ResNet',
    caltech256 = 'VGG', tinyimagenet = 'ResNet', voc = 'SSD', coco = 'SSD'
)


def get_strategies_object(methods: List[str], Masters: Dict[str, Master_Model], 
                          ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any], gtg_p: Dict[str, Any]) -> List[Any]:
    strategies = []
    for method in methods:
        if 'gtg' in method.split('_'):
            if method == 'gtg': strategies.append(dict_strategies[method]({**ct_p, 'Master_Model': Masters['M_None']}, t_p, al_p, gtg_p))
            else:
                am = gtg_p['am']
                del gtg_p['am']
                
                for a_matrix in am:
                    strategies.append(dict_strategies[method]({
                        **ct_p, 'Master_Model': Masters['M_GTG'] if method.split('_')[0] == 'lq' else Masters['M_LL']
                    }, t_p, al_p, {**gtg_p, 'am': a_matrix}))
                
        elif method == 'll' or method == 'tavaal':
            strategies.append(dict_strategies[method]({ **ct_p, 'Master_Model': Masters['M_LL'] }, t_p, al_p))
        else:
            strategies.append(dict_strategies[method]({**ct_p, 'Master_Model': Masters['M_None']}, t_p, al_p))
    
    return strategies


# to create a single master model for each type
def get_masters(methods: List[str], BBone: ResNet | SSD | VGG,
                ll_module_params: Dict[str, Any], gtg_module_params: Dict[str, Any],
                dataset_name: str) -> Dict[str, Master_Model]:
    
    LL_Mod, GTG_Mod, M_None = None, None, None
    ll, only_bb = False, False
    
    for method in methods:
        if (method == 'tavaal' or method.split('_')[0] == 'll') and not ll:
            LL_Mod = get_module('LL', ll_module_params)
            ll = True
        elif method == 'lq_gtg':
            GTG_Mod = get_module('GTG', (gtg_module_params, ll_module_params))
        elif not only_bb:
            M_None = Master_Model(BBone, None, dataset_name)
            only_bb = True
        else: continue
        
    Masters = { }
    # create and save the initial checkpoints of the masters
    if M_None != None: Masters['M_None'] = M_None
    if GTG_Mod != None: Masters['M_GTG'] = Master_Model(BBone, GTG_Mod, dataset_name)
    if LL_Mod != None: Masters['M_LL'] = Master_Model(BBone, LL_Mod, dataset_name)
    
    return Masters
    


def run_strategies(ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any], gtg_p: Dict[str, Any],
                   Masters: Dict[str, Master_Model], methods: List[str]) -> Tuple[Dict[str, Dict[str, List[float]]], List[int]]:

    results = { }
    n_lab_obs = [al_p['init_lab_obs'] + (iter * al_p['n_top_k_obs']) for iter in range(al_p['al_iters'])]
     
    # get the strategis object to run them
    strategies = get_strategies_object(methods, Masters, ct_p, t_p, al_p, gtg_p)
    
    ##########################################################################################################
    # START THE ACTIVE LEARNING PROECSS
    for strategy in strategies:
        logger.info(f'-------------------------- {strategy.strategy_name} --------------------------\n')
        results[strategy.strategy_name] = strategy.run()
    ##########################################################################################################
    
    return results, n_lab_obs               



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
        'gtg_t': args.gtg_tollerance,
        'gtg_i': args.gtg_iterations,
        
        'am': args.affinity_matrix,
        'am_s': args.affinity_matrix_strategy,
        'am_ts': args.affinity_matrix_threshold_strategy,
        'am_t': args.affinity_matrix_threshold,
        'e_s': args.entropy_strategy,
        
        'plb': args.perc_labeled_batch,
    }
    
        
    for dataset_name in args.datasets:
        
        if dataset_name in ['cifar10', 'cifar100', 'svhn', 'caltech256', 'tinyimagenet', 'fmnist']: 
            task = 'clf'
            task_params = cls_config
        else: 
            task = 'detection'
            task_params = det_config

        logger.info(f'----------------------- RUNNING ACTIVE LEARNING BENCHMARK ON {dataset_name} -----------------------\n')
        
        Dataset = get_dataset(task, dataset_name, init_lab_obs = al_params['init_lab_obs']) # get the dataset
        BBone = get_backbone(Dataset.n_classes, Dataset.n_channels, dict_backbone[dataset_name]) # the backbone is the same for all

        # create gtg dictionary parameters
        gtg_module_params = dict(
            **gtg_params, n_top_k_obs = al_params['n_top_k_obs'], 
            n_classes = Dataset.n_classes, 
            init_lab_obs = al_params['init_lab_obs'], 
            embedding_dim = BBone.get_rich_features_shape(),
            device = device
        )
        
        ll_module_params = get_ll_module_params(task, Dataset.image_size, dataset_name) # create learning loss dictionary parameters
            
        Masters = get_masters(args.methods, BBone, ll_module_params, gtg_module_params, dataset_name) # obtain the master models
                        
        logger.info('\n')
            
        common_training_params = {
            'Dataset': Dataset, 'device': device, 'timestamp': timestamp,
            'dataset_name': dataset_name, 'task': task, 'wandb_logs': args.wandb
        }
        
        for trial in range(args.trials):
            common_training_params['trial'] = trial
            
            logger.info(f'----------------------- SAMPLE ITERATION {trial + 1} / {args.trials} -----------------------\n')
            
            create_ts_dir(timestamp, dataset_name, str(trial))

            results, n_lab_obs = run_strategies(
                ct_p = common_training_params, t_p = task_params, al_p = al_params,
                gtg_p = gtg_params, Masters = Masters, methods = args.methods
            )
            
            plot_loss_curves(results, n_lab_obs, timestamp,
                             list(task_params['results_dict']['test'].keys()),
                             plot_png_name=f'{dataset_name}/{trial}/results.png')
            
        plot_res_std_mean(task, timestamp, dataset_name)


if __name__ == "__main__":
    
    start = time.time()
    main()
    end = time.time()
    
    logger.info(f'####################################################### TOTAL ELAPSED SECONDS: {end-start} #######################################################')
