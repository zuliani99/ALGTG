
import random

from strategies.ActiveLearner import ActiveLearner

from torch.utils.data import Subset

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class Random(ActiveLearner):
        
    def __init__(self, strategy_dict_params: Dict[str, Dict[str, Any] | bool]) -> None:
        self.method_name = f'{self.__class__.__name__}_LL' if strategy_dict_params['LL'] else self.__class__.__name__
        
        super().__init__(strategy_dict_params)
        
        
        
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        logger.info(f' => Sampling K unlabeled observations')
        if(len(sample_unlab_subset.indices) > n_top_k_obs):           
            sample = list(random.sample(list(sample_unlab_subset.indices), n_top_k_obs))
        else:
            sample = list(sample_unlab_subset.indices)
        logger.info(' DONE\n')
        return [list(sample_unlab_subset.indices).index(item) for item in sample], sample