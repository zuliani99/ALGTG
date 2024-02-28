
import torch
from torch.utils.data import DataLoader, Subset

from TrainEvaluate import TrainEvaluate
from Datasets import UniqueShuffle



class CoreSet(TrainEvaluate):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.LL = LL
        

    # X -> unlabeled X_set -> labeled
    def furthest_first(self, n_top_k_obs, unlabeled_indices):
        unlabeled_size = len(self.unlab_embeddings)
        
        if len(self.labeled_embeddings) == 0:
            min_dist = torch.full(unlabeled_size, float('inf'))
        else:
            dist_ctr = torch.cdist(self.unlab_embeddings, self.labeled_embeddings, p=2)
            # take the minimum for each row
            min_dist = torch.amin(dist_ctr, dim=1)
            
        overall_topk = []
        
        while len(overall_topk) < n_top_k_obs:
            idx = torch.argmax(min_dist).item()
            overall_topk.append(idx)
            print(idx, overall_topk)
            dist_new_ctr = torch.cdist(self.unlab_embeddings, self.unlab_embeddings[[idx], :], p=2)
            
            print(dist_new_ctr)
            
            for j in range(n_top_k_obs):
                min_dist[j] = min(min_dist[j].item(), dist_new_ctr[j, 0].item())

        return [unlabeled_indices[id].item() for id in overall_topk]
    
    
    

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
            
            # set the entire batch size to the dimension of the sampled unlabeled set
            self.unlab_batch_size = len(sample_unlab_subset)
                        
            self.unlab_train_dl = DataLoader(
                sample_unlab_subset, batch_size=self.unlab_batch_size,
                shuffle=True, pin_memory=True,
            )
            
            print(' => Getting the labeled and unlabeled embeddings')
            self.labeled_embeddings, _, _ = self.get_embeddings(self.lab_train_dl)
            self.unlab_embeddings, _, unlabeled_indices = self.get_embeddings(self.unlab_train_dl)
            print(' DONE\n')
                        
            print(' => Top K extraction using furthest_first')
            topk_idx_obs = self.furthest_first(n_top_k_obs, unlabeled_indices)
            print(' DONE\n')
            
            print(topk_idx_obs)
                        
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders(topk_idx_obs)
            print(' DONE\n')
            
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
            
        self.remove_model_opt()
            
        return results
        