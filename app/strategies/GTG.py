
import torch
from torch.utils.data import DataLoader, Subset

from ActiveLearner import ActiveLearner

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class GTG(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any], gtg_p) -> None:
        
        if gtg_p['am_ts'] != None:
            str_tresh_strat = f'{gtg_p['am_ts']}_{gtg_p['am_t']}' if gtg_p['am_ts'] != 'mean' else 'ts-mean'            
            strategy_name = f'{self.__class__.__name__}_{gtg_p['am_s']}_{gtg_p['am']}_{str_tresh_strat}_es-{gtg_p['e_s']}'
        else:
            strategy_name = f'{self.__class__.__name__}_{gtg_p['am_s']}_{gtg_p['am']}_es-{gtg_p['e_s']}'
                
        
        super().__init__(ct_p, t_p, al_p, strategy_name)
        
        if self.model.added_module != None: self.model.added_module.define_A_function(gtg_p['am'])
        
                
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
                
        logger.info(' => Evaluating unlabelled observations')
        embeds_dict = { 'module_out': torch.empty(0, dtype=torch.float32, device=torch.device('cpu')) }
        
        self.load_best_checkpoint()

        self.get_embeddings(self.unlab_train_dl, embeds_dict)
        
        logger.info(f' => Extracting the Top-k unlabelled observations')
        overall_topk = torch.topk(embeds_dict['module_out'], n_top_k_obs).indices.tolist()
        logger.info(' DONE\n')
        
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]