
import torch
from torch.utils.data import DataLoader

from torch.distributions import Categorical
import pdb

from strategies.Strategies import Strategies

from typing import Dict, Any, List



class BADGE(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], LL: bool) -> None:
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        
    
    
    def init_centers(self, n_top_k_obs: int) -> List[int]:
        ind = torch.argmax(torch.norm(self.embedds_dict['embedds'], dim=1))
        mu = self.embedds_dict['embedds'][ind].unsqueeze(0)
        indsAll = [ind.item()]
        centInds = [0] * len(self.embedds_dict['embedds'])
        cent = 0
        while len(mu) < n_top_k_obs:
            if len(mu) == 1:
                D2 = torch.cdist(self.embedds_dict['embedds'], mu).ravel().float()
            else:
                newD = torch.cdist(self.embedds_dict['embedds'], mu[-1].unsqueeze(0)).ravel().float()
                for i in range(len(self.embedds_dict['embedds'])):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            if torch.sum(D2) == 0.0: pdb.set_trace()
            Ddist = (D2 ** 2) / torch.sum(D2 ** 2)
            customDist = Categorical(Ddist)
            ind = customDist.sample()
            while ind.item() in indsAll:
                ind = customDist.sample()
            
            mu = torch.cat((mu, self.embedds_dict['embedds'][ind].unsqueeze(0)))
            
            indsAll.append(ind.item())
            cent += 1
            
        return indsAll



    def query(self, sample_unlab_subset: List[int], n_top_k_obs: int) -> List[int]:
                        
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True
        )
            
        print(' => Getting the unlabeled embeddings')
        self.embedds_dict = {'embedds': None, 'idxs': None}
        self.get_embeddings(self.unlab_train_dl, self.embedds_dict)
        print(' DONE\n')
                        
        print(' => Top K extraction using init_centers')
        topk_idx_obs = self.init_centers(n_top_k_obs)
        print(' DONE\n')
        
        self.clear_cuda_variables([self.embedds_dict])
        
        return [self.embedds_dict['idxs'][id].item() for id in topk_idx_obs]

    