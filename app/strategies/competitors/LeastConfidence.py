
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from strategies.Strategies import Strategies

from typing import Dict, Any, List



class LeastConfidence(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], LL: bool, al_iters: int, n_top_k_obs: int, unlab_sample_dim: int) -> None:
        super().__init__(al_params, LL, al_iters, n_top_k_obs, unlab_sample_dim)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> List[int]:
                                
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True
        )
            
        print(' => Getting the unlabeled probebilities')
        self.embedds_dict = {'probs': None, 'idxs': None}
        self.get_embeddings(self.unlab_train_dl, self.embedds_dict)
        unlab_probs = F.softmax(self.embedds_dict['probs'], dim=1)
        print(' DONE\n')
            
        topk_idx_obs = torch.topk(unlab_probs.max(1)[0], n_top_k_obs)
        
                
        return [self.embedds_dict['idxs'][id].item() for id in topk_idx_obs.indices.tolist()]
    