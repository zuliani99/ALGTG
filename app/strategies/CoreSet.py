
import torch
from torch.utils.data import DataLoader, Subset

from TrainEvaluate import TrainEvaluate


class CoreSet(TrainEvaluate):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.LL = LL
    

    
    def furthest_first(self, n_top_k_obs):
        unlabeled_size = self.unlab_embeddings.size(0)
        if self.labeled_embeddings.size(0) == 0:
            min_dist = float('inf') * torch.ones(unlabeled_size)
        else:
            dist_ctr = torch.cdist(self.unlab_embeddings, self.labeled_embeddings)
            min_dist, _ = torch.min(dist_ctr, dim=1)
        
        overall_topk = []

        while len(overall_topk) < n_top_k_obs:
            idx = torch.argmax(min_dist)
            overall_topk.append(idx.item())
            dist_new_ctr = torch.cdist(self.unlab_embeddings, self.unlab_embeddings[idx].unsqueeze(0))
            min_dist = torch.min(min_dist, dist_new_ctr[:, 0])

        return overall_topk

    
    
    

    def run(self, al_iters, epochs, unlab_sample_dim, n_top_k_obs):
        
        iter = 1
        
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, n_top_k_obs, iter, results)
        
        # start of the loop
        while len(self.unlabeled_indices) > 0 and iter < al_iters:
            iter += 1
            
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
                                        
            sample_unlab_subset = Subset(
                self.non_transformed_trainset,
                self.get_unlabebled_samples(unlab_sample_dim, iter)
            )
            
            # set the entire batch size to the dimension of the sampled unlabeled set
            self.unlab_train_dl = DataLoader(
                sample_unlab_subset, batch_size=self.batch_size,
                shuffle=True, pin_memory=True,
            )
            
            print(' => Getting the labeled and unlabeled embeddings')
            _, self.labeled_embeddings, _ = self.get_embeddings(self.lab_train_dl)
            unlabeled_indices, self.unlab_embeddings, _ = self.get_embeddings(self.unlab_train_dl)
            print(' DONE\n')
                        
            print(' => Top K extraction')
            topk_idx_obs = self.furthest_first(n_top_k_obs, unlabeled_indices)
            print(' DONE\n')
            
                        
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders([unlabeled_indices[id].item() for id in topk_idx_obs])
            print(' DONE\n')
            
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
            
        self.remove_model_opt()
            
        return results
        