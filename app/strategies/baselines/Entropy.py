
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from strategies.Strategies import Strategies
from utils import entropy

from typing import Dict, Any, List


class Entropy(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], LL: bool, al_iters: int, n_top_k_obs: int, unlab_sample_dim: int) -> None:
        super().__init__(al_params, LL, al_iters, n_top_k_obs, unlab_sample_dim)
                
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        
        
        
    def query(self, sample_unlab_subset: List[int], n_top_k_obs: int) -> List[int]:
        
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset,
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
                
        print(' => Evalueting unlabeled observations')
        embeds_dict = {'probs': None, 'idxs': None}
        self.get_embeddings(self.unlab_train_dl, embeds_dict)
        prob_dist = F.softmax(embeds_dict['probs'], dim=1)
        print(' DONE\n')
                
        tot_entr = entropy(prob_dist).to(self.device)
        overall_topk = torch.topk(tot_entr, n_top_k_obs)
        
        
        return [embeds_dict['idxs'][id].item() for id in overall_topk.indices.tolist()]