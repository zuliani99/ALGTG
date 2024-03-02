
import random

from strategies.Strategies import Strategies


class Random(Strategies):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        
        
    def query(self, sample_unlab_subset, n_top_k_obs):         
        if(len(sample_unlab_subset.indices) > n_top_k_obs):           
            return random.sample(sample_unlab_subset.indices, n_top_k_obs)
        else:
            return sample_unlab_subset.indices