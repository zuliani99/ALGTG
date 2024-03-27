
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from strategies.Strategies import Strategies

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)



class CDAL(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], training_params: Dict[str, Any], LL: bool) -> None:
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__

        super().__init__(al_params, training_params, LL)
        


    def furthest_first(self, X: torch.Tensor, X_set: torch.Tensor, n_top_k_obs: int) -> List[int]:
        m = X.size(0)
        if X_set.size(0) == 0:
            min_dist = float('inf') * torch.ones(m)
        else:
            dist_ctr = torch.cdist(X, X_set)
            min_dist, _ = torch.min(dist_ctr, dim=1)
        
        overall_topk = []

        while len(overall_topk) < n_top_k_obs:
            idx = torch.argmax(min_dist)
            overall_topk.append(idx.item())
            dist_new_ctr = torch.cdist(X, X[idx].unsqueeze(0))
            min_dist = torch.min(min_dist, dist_new_ctr[:, 0])

        return overall_topk


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
                                
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True
        )
            
        logger.info(' => Getting the labeled and unlabeled probebilities')
        self.lab_embedds_dict, self.unlab_embedds_dict = {'probs': None}, {'probs': None}
        self.get_embeddings(self.lab_train_dl, self.lab_embedds_dict)
        self.get_embeddings(self.unlab_train_dl, self.unlab_embedds_dict)
        
        lab_probs = F.softmax(self.lab_embedds_dict['probs'], dim=1)
        unlab_probs = F.softmax(self.unlab_embedds_dict['probs'], dim=1)
        
        logger.info(' DONE\n')
        
        topk_idx_obs = self.furthest_first(unlab_probs, lab_probs, n_top_k_obs)
                    
        
        return topk_idx_obs, [self.unlab_embedds_dict['idxs'][id].item() for id in topk_idx_obs]
    