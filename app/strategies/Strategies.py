
from train_evaluate.TrainEvaluate import TrainEvaluate

from torch.utils.data import Subset
import torch

from typing import List, Dict, Any
import random
import copy
import math


class Strategies(TrainEvaluate):
    def __init__(self, al_params: Dict[str, Any], LL: bool, al_iters: int, n_top_k_obs: int, unlab_sample_dim: int) -> None:
        
        super().__init__(al_params, LL)
        self.al_iters = al_iters
        self.n_top_k_obs = n_top_k_obs
        self.unlab_sample_dim = unlab_sample_dim
        self.samp_iter = al_params['samp_iter']
        self.dataset_id = al_params['DatasetChoice'].dataset_id
        self.get_sampled_sets()
                
                
                
    def get_sampled_sets(self):
        self.sampled_list = []
        
        temp_unlabeled_indices = copy.deepcopy(self.unlabeled_indices)
        
        limit = int(math.ceil((self.al_iters * self.n_top_k_obs) / self.unlab_sample_dim))
            
        print('limit', limit)
        print('random seed', self.dataset_id * self.samp_iter)
        random.seed(self.dataset_id * self.samp_iter)
        
        for idx in range(limit):
            if(len(temp_unlabeled_indices) > self.unlab_sample_dim):
                seq = random.sample(temp_unlabeled_indices, self.unlab_sample_dim)
            else: 
                seq = temp_unlabeled_indices
            
            print(idx, seq[-5:])
            self.sampled_list.append(seq)
            for x in seq: temp_unlabeled_indices.remove(x)
        
        print('self.sampled_list len', len(self.sampled_list))
        random.seed(10001)
            
        
        
    def run(self, epochs: int) -> Dict[str, List[float]]:
        
        self.iter = 1
        
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        print(f'----------------------- ITERATION {self.iter} / {self.al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, self.n_top_k_obs, self.iter, results)
        
        # start of the loop
        while len(self.unlabeled_indices) > 0 and self.iter < self.al_iters:
            idx_list = (self.iter * self.n_top_k_obs) // self.unlab_sample_dim
            
            self.iter += 1
            
            print(f'----------------------- ITERATION {self.iter} / {self.al_iters} -----------------------\n')
            
            print(f' => Working with the unlabeled sampled list {idx_list}')
            sample_unlab_subset = Subset(self.non_transformed_trainset, self.sampled_list[idx_list])
            print(' START QUERY PROCESS\n')
            
            
            # run method query strategy
            topk_idx_obs = self.query(sample_unlab_subset, self.n_top_k_obs)
                    
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders(topk_idx_obs, idx_list)
            print(' DONE\n')

            # iter + 1
            self.train_evaluate_save(epochs, self.iter * self.n_top_k_obs, self.iter, results)
                        
                        
        #self.remove_model_opt()
        #self.clear_cuda_variables([self.model])
        del self.model
        torch.cuda.empty_cache()
            
        return results