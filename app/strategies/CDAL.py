
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from strategies.Strategies import Strategies



class CDAL(Strategies):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__


    def furthest_first(self, X, X_set, n_top_k_obs):
        m = X.size(0)
        if X_set.size(0) == 0:
            min_dist = float('inf') * torch.ones(m)
        else:
            dist_ctr = torch.cdist(X, X_set)
            min_dist, _ = torch.min(dist_ctr, dim=1)
        
        overall_topk = []

        while len(overall_topk) < n_top_k_obs:
            idx = torch.argmax(min_dist)
            overall_topk.append(idx.item())
            dist_new_ctr = torch.cdist(X, X[idx].unsqueeze(0))
            min_dist = torch.min(min_dist, dist_new_ctr[:, 0])

        return overall_topk


    def query(self, sample_unlab_subset, n_top_k_obs):
                                
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True
        )
            
        print(' => Getting the labeled and unlabeled probebilities')
        self.lab_embedds_dict, self.unlab_embedds_dict = {'probs': None}, {'probs': None}
        self.get_embeddings(self.lab_train_dl, self.lab_embedds_dict)
        self.get_embeddings(self.unlab_train_dl, self.unlab_embedds_dict)
        
        lab_probs = F.softmax(self.lab_embedds_dict['probs'], dim=1)
        unlab_probs = F.softmax(self.unlab_embedds_dict['probs'], dim=1)
        
        print(' DONE\n')
        
        topk_idx_obs = self.furthest_first(unlab_probs, lab_probs, n_top_k_obs)
                    
        self.clear_cuda_variables([self.embedds_dict])
        
        return [self.embedds_dict['idxs'][id].item() for id in topk_idx_obs.indices.tolist()]
    