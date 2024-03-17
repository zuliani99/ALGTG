
from train_evaluate.TrainEvaluate import TrainEvaluate

from torch.utils.data import Subset

from typing import List, Dict, Any
import random
import math




class Strategies(TrainEvaluate):
    def __init__(self, al_params: Dict[str, Any], training_params: Dict[str, Any], LL: bool) -> None:
        
        super().__init__(training_params, LL)
        self.al_iters: int = al_params['al_iters']
        self.n_top_k_obs: int = al_params['n_top_k_obs']
        self.unlab_sample_dim: int = al_params['unlab_sample_dim']
        
        self.samp_iter: int = training_params['samp_iter']
        self.dataset_id: int = training_params['DatasetChoice'].dataset_id
        self.get_sampled_sets()
                
                
                
    def get_sampled_sets(self):
        self.subset_sampled_list : List[Subset] = []
                
        limit = int(math.ceil((self.al_iters * self.n_top_k_obs) / self.unlab_sample_dim))
            
        print('limit', limit)
        print('random seed', self.dataset_id * self.samp_iter)
        random.seed(self.dataset_id * self.samp_iter)
        
        for idx in range(limit):
            if(len(self.unlabeled_indices) > self.unlab_sample_dim):
                seq = random.sample(self.unlabeled_indices, self.unlab_sample_dim)
            else: 
                seq = self.unlabeled_indices
            
            print(idx, seq[-5:])
            self.subset_sampled_list.append(Subset(self.non_transformed_trainset, seq))
            for x in seq: self.unlabeled_indices.remove(x)
        
        print('self.subset_sampled_list len', len(self.subset_sampled_list))
        random.seed(10001) # reset the random seed
        del self.unlabeled_indices
        
    
    def check_iterated_subset_sampled_list(self) -> bool:
        return all(not subset.indices for subset in self.subset_sampled_list)
        
        
    def run(self, epochs: int) -> Dict[str, List[float]]:
        
        self.iter = 1
        
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        print(f'----------------------- ITERATION {self.iter} / {self.al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, self.n_top_k_obs, self.iter, results)
        
        
        # start of the loop
        while not self.check_iterated_subset_sampled_list() and self.iter < self.al_iters:
            
            idx_list = (self.iter * self.n_top_k_obs) // self.unlab_sample_dim
            
            self.iter += 1
            
            print(f'----------------------- ITERATION {self.iter} / {self.al_iters} -----------------------\n')
            
            print(f' => Working with the unlabeled sampled list {idx_list}')
            #sample_unlab_subset = Subset(self.non_transformed_trainset, self.sampled_list[idx_list])
            print(' START QUERY PROCESS\n')
            
            # run method query strategy
            topk_idx_obs = self.query(self.subset_sampled_list[idx_list], self.n_top_k_obs)
                    
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.update_sets(topk_idx_obs, idx_list)
            print(' DONE\n')

            # iter + 1
            self.train_evaluate_save(epochs, self.iter * self.n_top_k_obs, self.iter, results)
            
            
        return results