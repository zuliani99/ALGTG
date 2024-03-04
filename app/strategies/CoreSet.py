
import torch
from torch.utils.data import DataLoader

from strategies.Strategies import Strategies


class CoreSet(Strategies):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
    

    
    def furthest_first(self, n_top_k_obs):
        unlabeled_size = self.unlab_embedds_dict['embedds'].size(0)
        if self.lab_embedds_dict['embedds'].size(0) == 0:
            min_dist = float('inf') * torch.ones(unlabeled_size)
        else:
            dist_ctr = torch.cdist(self.unlab_embedds_dict['embedds'], self.lab_embedds_dict['embedds'])
            min_dist, _ = torch.min(dist_ctr, dim=1)
        
        overall_topk = []

        while len(overall_topk) < n_top_k_obs:
            idx = torch.argmax(min_dist)
            overall_topk.append(idx.item())
            dist_new_ctr = torch.cdist(self.unlab_embedds_dict['embedds'],
                                       self.unlab_embedds_dict['embedds'][idx].unsqueeze(0))
            min_dist = torch.min(min_dist, dist_new_ctr[:, 0])

        return overall_topk


    
    def query(self, sample_unlab_subset, n_top_k_obs):
            
        # set the entire batch size to the dimension of the sampled unlabeled set
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True,
        )
            
        print(' => Getting the labeled and unlabeled embeddings')
        self.lab_embedds_dict = {'embedds': None}
        self.unlab_embedds_dict = {'embedds': None, 'idxs': None}
            
        self.get_embeddings(self.lab_train_dl, self.lab_embedds_dict)
        self.get_embeddings(self.unlab_train_dl, self.unlab_embedds_dict)
        print(' DONE\n')
                        
        print(' => Top K extraction')
        topk_idx_obs = self.furthest_first(n_top_k_obs)
        print(' DONE\n')
        
        self.clear_cuda_variables([self.lab_embedds_dict, self.unlab_embedds_dict])
        
        return [self.unlab_embedds_dict['idxs'][id].item() for id in topk_idx_obs]
    