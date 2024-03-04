
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from strategies.Strategies import Strategies


class LeastConfidence(Strategies):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__


    def query(self, sample_unlab_subset, n_top_k_obs):
                                
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True
        )
            
        print(' => Getting the unlabeled probebilities')
        self.embedds_dict = {'embedds': None, 'idxs': None}
        self.get_embeddings(self.unlab_train_dl, self.embedds_dict)
        unlab_probs = F.softmax(self.embedds_dict['embedds'], dim=1)
        print(' DONE\n')
            
        topk_idx_obs = torch.topk(unlab_probs.max(1)[0], n_top_k_obs)
        
        self.clear_cuda_variables([self.embedds_dict])
        
        return [self.embedds_dict['idxs'][id].item() for id in topk_idx_obs.indices.tolist()]
    