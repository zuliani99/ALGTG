
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ...strategies.ActiveLearner import ActiveLearner

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class LeastConfidence(ActiveLearner):
    
    def __init__(self, strategy_dict_params: Dict[str, Dict[str, Any] | bool]) -> None:
        self.method_name = f'{self.__class__.__name__}_LL' if strategy_dict_params['LL'] else self.__class__.__name__
        
        super().__init__(strategy_dict_params)
        


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
                                
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.ct_p['batch_size'],
            shuffle=False, pin_memory=True
        )
            
        logger.info(' => Getting the unlabeled probebilities')
        self.embedds_dict = {
            'probs': torch.empty((0, self.dataset.n_classes), dtype=torch.float32), 
            'idxs': torch.empty(0, dtype=torch.int8)
        }
        self.get_embeddings(self.unlab_train_dl, self.embedds_dict)
        unlab_probs = F.softmax(self.embedds_dict['probs'], dim=1)
        logger.info(' DONE\n')
            
        topk_idx_obs = torch.topk(unlab_probs.max(1)[0], n_top_k_obs)
        
                
        return topk_idx_obs.indices.tolist(), [int(self.embedds_dict['idxs'][id].item()) for id in topk_idx_obs.indices.tolist()]
    