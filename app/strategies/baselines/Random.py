
import random

from strategies.Strategies import Strategies

from torch.utils.data import Subset

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class Random(Strategies):
        
    def __init__(self, al_params: Dict[str, Any], training_params: Dict[str, Any], LL: bool) -> None:
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        
        super().__init__(al_params, training_params, LL)
        
        
        
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        logger.info(f' => Sampling K unlabeled observations')
        if(len(sample_unlab_subset.indices) > n_top_k_obs):           
            sample = list(random.sample(list(sample_unlab_subset.indices), n_top_k_obs))
        else:
            sample = list(sample_unlab_subset.indices)
        logger.info(' DONE\n')
        return [list(sample_unlab_subset.indices).index(item) for item in sample], sample