
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from strategies.Strategies import Strategies
from utils import entropy


class Entropy(Strategies):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
                
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        
        
        
    def query(self, sample_unlab_subset, n_top_k_obs):
        
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset,
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
                
        print(' => Evalueting unlabeled observations')
        embeds_dict = {'embedds': None, 'idxs': None}
        self.get_embeddings(self.unlab_train_dl, embeds_dict)
        prob_dist = F.softmax(embeds_dict['embedds'], dim=1)
        print(' DONE\n')
                
        tot_entr = entropy(prob_dist).to(self.device)
        overall_topk = torch.topk(tot_entr, n_top_k_obs)
        
        self.clear_cuda_variables([embeds_dict, tot_entr])
        
        return [embeds_dict['idxs'][id].item() for id in overall_topk.indices.tolist()]