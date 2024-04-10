
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from strategies.Strategies import Strategies

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class LearningLossStrategy(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], training_params: Dict[str, Any], LL: bool) -> None:
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        
        super().__init__(al_params, training_params, LL)
                
                
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset,
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
                
        logger.info(' => Evaluating unlabeled observations')
        embeds_dict = {
            'out_weird': torch.empty(0, dtype=torch.float32),
            'idxs': torch.empty(0, dtype=torch.int8)
        }
        self.get_embeddings(self.unlab_train_dl, embeds_dict)
        
        logger.info(f' => Extracting the Top-k unlabeled observations')
        overall_topk = torch.topk(embeds_dict['out_weird'], n_top_k_obs)
        logger.info(' DONE\n')
        
        return overall_topk.indices.tolist(), [int(embeds_dict['idxs'][id].item()) for id in overall_topk.indices.tolist()]