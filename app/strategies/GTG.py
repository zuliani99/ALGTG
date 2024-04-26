
import torch
from torch.utils.data import DataLoader, Subset

from ActiveLearner import ActiveLearner

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class GTG(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any], gtg_p) -> None:
        
        str_rbf = 'rbf_' if gtg_p['rbf_aff'] else ''
        if gtg_p['threshold_strategy'] != None:
            str_treshold = f'{gtg_p['threshold_strategy']}_{gtg_p['threshold']}' if gtg_p['threshold_strategy'] != 'mean' else 'mean'            
            strategy_name = f'{self.__class__.__name__}_{gtg_p['strategy_type']}_{str_rbf}{gtg_p['A_function']}_{gtg_p['ent_strategy'].name}_{str_treshold}'
        else:
            strategy_name = f'{self.__class__.__name__}_{gtg_p['strategy_type']}_{str_rbf}{gtg_p['A_function']}_{gtg_p['ent_strategy'].name}'
                
        
        super().__init__(ct_p, t_p, al_p, strategy_name)
        
        if self.model.added_module != None:
            self.model.added_module.define_additional_parameters(gtg_p)
        
                
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset,
            batch_size=self.ds_t_p['batch_size'], shuffle=False, pin_memory=True
        )
                
        logger.info(' => Evaluating unlabeled observations')
        embeds_dict = {
            'module_out': torch.empty(0, dtype=torch.float32),
            'idxs': torch.empty(0, dtype=torch.int8)
        }
        
        self.load_best_checkpoint()

        self.get_embeddings(self.unlab_train_dl, embeds_dict)
        
        logger.info(f' => Extracting the Top-k unlabeled observations')
        overall_topk = torch.topk(embeds_dict['module_out'], n_top_k_obs)
        logger.info(' DONE\n')
        
        return overall_topk.indices.tolist(), [int(embeds_dict['idxs'][id].item()) for id in overall_topk.indices.tolist()]