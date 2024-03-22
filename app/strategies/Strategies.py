
from train_evaluate.TrainEvaluate import TrainEvaluate

from torch.utils.data import Subset

from typing import List, Dict, Any
import random
import math
import copy




class Strategies(TrainEvaluate):
    def __init__(self, al_params:  Dict[str, Any], training_params: Dict[str, Any], LL: bool) -> None:
        
        super().__init__(training_params, LL)
        self.al_iters: int = al_params['al_iters']
        self.n_top_k_obs: int = al_params['n_top_k_obs']
        self.unlab_sample_dim: int = al_params['unlab_sample_dim']
        
        self.unlabeled_indices: List[int] = copy.deepcopy(training_params['DatasetChoice'].unlabeled_indices)
        
        self.get_sampled_sets()
                
                
                
    def get_sampled_sets(self) -> None:
        self.unlab_sampled_list : List[Subset] = []
                
        limit = int(math.ceil((self.al_iters * self.n_top_k_obs) / self.unlab_sample_dim))
            
        print(f'Number of subsets of dimension {self.unlab_sample_dim}: {limit}')
        print(f'Random seed for reproducibility: {self.dataset_id * self.samp_iter}')
        random.seed(self.dataset_id * self.samp_iter)
        
        for idx in range(limit):
            if(len(self.unlabeled_indices) > self.unlab_sample_dim):
                seq = random.sample(self.unlabeled_indices, self.unlab_sample_dim)
            else: 
                seq = self.unlabeled_indices
            
            print(f'Index: {idx} \t Last 5 observations: {seq[-5:]}')
            
            unlabeled_subset = Subset(self.non_transformed_trainset, seq)
            d_labels = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
            
            for _, _, lab in unlabeled_subset: d_labels[str(lab)] += 1
            
            self.unlab_sampled_list.append(unlabeled_subset)
            for x in seq: self.unlabeled_indices.remove(x)
            print(d_labels)
        
        print(f'Lenght of the unlabeled sampled subset list: {len(self.unlab_sampled_list)}')
        random.seed(10001) # reset the random seed
        del self.unlabeled_indices
        
        
    
    # check if all the subsets have been viewed
    def check_iterated_unlab_sampled_list(self) -> bool:
        return all(not subset.indices for subset in self.unlab_sampled_list)
        
        
        
    def run(self, epochs: int) -> Dict[str, List[float]]:
        
        self.iter = 1
        
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        print(f'----------------------- ITERATION {self.iter} / {self.al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, self.n_top_k_obs, self.iter, results)
        
        
        # start of the loop
        while not self.check_iterated_unlab_sampled_list() and self.iter < self.al_iters:
            
            # seting the indices of the subset list
            idx_list = (self.iter * self.n_top_k_obs) // self.unlab_sample_dim
            
            self.iter += 1
            
            print(f'----------------------- ITERATION {self.iter} / {self.al_iters} -----------------------\n')
            
            print(f' => Working with the unlabeled sampled list {idx_list}')
            print(' START QUERY PROCESS\n')
            
            # run method query strategy
            topk_idx_obs: List[int] = self.query(self.unlab_sampled_list[idx_list], self.n_top_k_obs)
            
            label_topk = [self.transformed_trainset[k][2] for k in topk_idx_obs]
            d_labels = {}
            for cls in self.classes: d_labels[cls] = 0
            keys_d_labels = list(d_labels.keys())
            for l in label_topk: d_labels[keys_d_labels[l]] += 1
            print('Number of observations per class added to the labeled set:')
            print(d_labels)
            
            
            # get the new labeled indices from the subset to plot the tsne embeddings
            indices_unlab = [self.unlab_sampled_list[idx_list].indices.index(item) for item in topk_idx_obs]
            
            # Saving the tsne embeddings plot
            self.save_tsne(idx_list, indices_unlab, self.iter)

            # modify the datasets and dataloader and plot the tsne
            self.update_sets(topk_idx_obs, idx_list)

            # iter + 1
            self.train_evaluate_save(epochs, self.iter * self.n_top_k_obs, self.iter, results)
            
            
        return results