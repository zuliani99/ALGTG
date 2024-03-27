
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from strategies.Strategies import Strategies
from utils import entropy

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class Entropy(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], training_params: Dict[str, Any], LL: bool) -> None:
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        
        super().__init__(al_params, training_params, LL)
                
        
        
        
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int]]:
        
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset,
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
                
        logger.info(' => Evaluating unlabeled observations')
        embeds_dict = {'probs': None, 'idxs': None}
        self.get_embeddings(self.unlab_train_dl, embeds_dict)
        prob_dist = F.softmax(embeds_dict['probs'], dim=1)
        logger.info(' DONE\n')

        logger.info(f' => Extracting the Top-k unlabeled observations')
        tot_entr = entropy(prob_dist).to(self.device)
        overall_topk = torch.topk(tot_entr, n_top_k_obs)
        logger.info(' DONE\n')
        
        return overall_topk.indices.tolist(), [embeds_dict['idxs'][id].item() for id in overall_topk.indices.tolist()]