
import torch
from torch.utils.data import DataLoader, Subset

from strategies.Strategies import Strategies

from typing import Dict, Any, List

import logging
logger = logging.getLogger(__name__)


class CoreSet(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], training_params: Dict[str, Any], LL: bool) -> None:
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__

        super().__init__(al_params, training_params, LL)
        
    

    
    def furthest_first(self, n_top_k_obs: int) -> List[int]:
        unlabeled_size = self.unlab_embedds_dict['embedds'].size(0)
        if self.lab_embedds_dict['embedds'].size(0) == 0:
            min_dist = float('inf') * torch.ones(unlabeled_size)
        else:
            dist_ctr = torch.cdist(self.unlab_embedds_dict['embedds'], self.lab_embedds_dict['embedds'])
            min_dist, _ = torch.min(dist_ctr, dim=1)
        
        overall_topk = []

        while len(overall_topk) < n_top_k_obs:
            idx = torch.argmax(min_dist)
            overall_topk.append(idx.item())
            dist_new_ctr = torch.cdist(self.unlab_embedds_dict['embedds'],
                                       self.unlab_embedds_dict['embedds'][idx].unsqueeze(0))
            min_dist = torch.min(min_dist, dist_new_ctr[:, 0])

        return overall_topk


    
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> List[int]:
            
        # set the entire batch size to the dimension of the sampled unlabeled set
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True,
        )
            
        logger.info(' => Getting the labeled and unlabeled embeddings')
        self.lab_embedds_dict = {'embedds': None}
        self.unlab_embedds_dict = {'embedds': None, 'idxs': None}
            
        self.get_embeddings(self.lab_train_dl, self.lab_embedds_dict)
        self.get_embeddings(self.unlab_train_dl, self.unlab_embedds_dict)
        logger.info(' DONE\n')
                        
        logger.info(' => Top K extraction')
        topk_idx_obs = self.furthest_first(n_top_k_obs)
        logger.info(' DONE\n')
        
        del self.lab_embedds_dict
        torch.cuda.empty_cache()
        
        return [self.unlab_embedds_dict['idxs'][id].item() for id in topk_idx_obs]
    