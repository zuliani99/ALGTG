
import torch
from torch.utils.data import DataLoader, Subset

from ActiveLearner import ActiveLearner

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class LearningLoss(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any]) -> None:
       
        name = self.__class__.__name__
        if 'll_version' in ct_p and ct_p['ll_version'] == 2: name += '_v2'
            
        super().__init__(ct_p, t_p, al_p, name)
                
                
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