
import random

from strategies.Strategies import Strategies

from typing import Dict, Any, List


class Random(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], LL: bool) -> None:
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        
        
    def query(self, sample_unlab_subset: List[int], n_top_k_obs: int) -> List[int]:
        if(len(sample_unlab_subset.indices) > n_top_k_obs):           
            return random.sample(sample_unlab_subset.indices, n_top_k_obs)
        else:
            return sample_unlab_subset.indices