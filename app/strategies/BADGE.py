
import torch
from torch.utils.data import DataLoader, Subset

#from scipy import stats
from torch.distributions import Categorical
import pdb

from TrainEvaluate import TrainEvaluate


class BADGE(TrainEvaluate):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.LL = LL
        
    
    def init_centers(self, n_top_k_obs):
        ind = torch.argmax(torch.norm(self.unlab_embeddings, dim=1))
        mu = self.unlab_embeddings[ind].unsqueeze(0)
        indsAll = [ind.item()]
        centInds = [0] * len(self.unlab_embeddings)
        cent = 0
        while len(mu) < n_top_k_obs:
            if len(mu) == 1:
                D2 = torch.cdist(self.unlab_embeddings, mu).ravel().float()
            else:
                newD = torch.cdist(self.unlab_embeddings, mu[-1].unsqueeze(0)).ravel().float()
                for i in range(len(self.unlab_embeddings)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            if torch.sum(D2) == 0.0: pdb.set_trace()
            Ddist = (D2 ** 2) / torch.sum(D2 ** 2)
            customDist = Categorical(Ddist)
            ind = customDist.sample()
            while ind.item() in indsAll:
                ind = customDist.sample()
            
            mu = torch.cat((mu, self.unlab_embeddings[ind].unsqueeze(0)))
            
            indsAll.append(ind.item())
            cent += 1
            
        return indsAll
    

    def run(self, al_iters, epochs, unlab_sample_dim, n_top_k_obs):
        
        iter = 1
        
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, n_top_k_obs, iter, results)
        
        # start of the loop
        while len(self.unlab_train_subset) > 0 and iter < al_iters:
            iter += 1
            
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
                                        
            sample_unlab_subset = Subset(
                self.non_transformed_trainset,
                self.get_unlabebled_samples(unlab_sample_dim, iter)
            )
                        
            self.unlab_train_dl = DataLoader(
                sample_unlab_subset, batch_size=self.batch_size,
                shuffle=True, pin_memory=True
            )
            
            print(' => Getting the unlabeled embeddings')
            self.unlab_embeddings, _,  unlabeled_indices = self.get_embeddings(self.unlab_train_dl)
            print(' DONE\n')
                        
            print(' => Top K extraction using init_centers')
            indices_to_convert = self.init_centers(n_top_k_obs)
            print(' DONE\n')
                                    
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders([unlabeled_indices[id].item() for id in indices_to_convert])
            print(' DONE\n')
            
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
            
        self.remove_model_opt()
            
        return results
        