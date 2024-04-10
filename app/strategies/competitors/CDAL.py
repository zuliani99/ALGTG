
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ActiveLearner import ActiveLearner

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)



class CDAL(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any]) -> None:
        self.method_name = self.__class__.__name__
        
        super().__init__(ct_p, t_p, al_p)
        


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
            sample_unlab_subset, batch_size=self.t_p['batch_size'],
            shuffle=False, pin_memory=True
        )
            
        logger.info(' => Getting the labeled and unlabeled probebilities')
        lab_embedds_dict = {'probs': torch.empty((0, self.dataset.n_classes), dtype=torch.float32)}
        unlab_embedds_dict = {'probs': torch.empty((0, self.dataset.n_classes), dtype=torch.float32)}
        self.get_embeddings(self.lab_train_dl, lab_embedds_dict)
        self.get_embeddings(self.unlab_train_dl, unlab_embedds_dict)
        
        lab_probs = F.softmax(lab_embedds_dict['probs'], dim=1)
        unlab_probs = F.softmax(unlab_embedds_dict['probs'], dim=1)
        
        logger.info(' DONE\n')
        
        topk_idx_obs = self.furthest_first(unlab_probs, lab_probs, n_top_k_obs)
                    
        
        return topk_idx_obs, [int(unlab_embedds_dict['idxs'][id].item()) for id in topk_idx_obs]
    