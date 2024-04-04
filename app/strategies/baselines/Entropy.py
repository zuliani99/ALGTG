
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from ...strategies.ActiveLearner import ActiveLearner
from ...utils import entropy

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class Entropy(ActiveLearner):
    
    def __init__(self, strategy_dict_params: Dict[str, Dict[str, Any] | bool]) -> None:
        self.method_name = f'{self.__class__.__name__}_LL' if strategy_dict_params['LL'] else self.__class__.__name__
        
        super().__init__(strategy_dict_params)
                
        
        
        
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset,
            batch_size=self.ct_p['batch_size'], shuffle=False, pin_memory=True
        )
                
        logger.info(' => Evaluating unlabeled observations')
        embeds_dict = {
            'probs': torch.empty((0, self.dataset.n_classes), dtype=torch.float32),
            'idxs': torch.empty(0, dtype=torch.int8)
        }
        self.get_embeddings(self.unlab_train_dl, embeds_dict)
        prob_dist = F.softmax(embeds_dict['probs'], dim=1)
        logger.info(' DONE\n')

        logger.info(f' => Extracting the Top-k unlabeled observations')
        tot_entr = entropy(prob_dist).to(self.device)
        overall_topk = torch.topk(tot_entr, n_top_k_obs)
        logger.info(' DONE\n')
        
        return overall_topk.indices.tolist(), [int(embeds_dict['idxs'][id].item()) for id in overall_topk.indices.tolist()]