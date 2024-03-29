
from utils import count_class_observation
from train_evaluate.TrainEvaluate import TrainEvaluate

from torch.utils.data import Subset

from typing import List, Dict, Any
import random
import math
import copy

import logging
logger = logging.getLogger(__name__)




class Strategies(TrainEvaluate):
    def __init__(self, al_params:  Dict[str, Any], training_params: Dict[str, Any], LL: bool) -> None:
        
        self.iter = 1
        
        super().__init__(training_params, LL)
        
        self.al_iters: int = al_params['al_iters']
        self.n_top_k_obs: int = al_params['n_top_k_obs']
        self.unlab_sample_dim: int = al_params['unlab_sample_dim']
        
        self.unlabeled_indices: List[int] = copy.deepcopy(training_params['DatasetChoice'].unlabeled_indices)
        
        logger.info(' => Obtaining the intial sample of unlabeled observations...')
        self.get_sampled_sets()
        logger.info(' DONE\n')
        
        
                
    def get_sampled_sets(self) -> None:
        self.unlab_sampled_list : List[Subset] = []
                
        limit = int(math.ceil((self.al_iters * self.n_top_k_obs) / self.unlab_sample_dim))
            
        logger.info(f'Number of subsets of dimension {self.unlab_sample_dim}: {limit}')
        logger.info(f'Random seed for reproducibility: {self.dataset_id * self.samp_iter}')
        random.seed(self.dataset_id * self.samp_iter)
        
        for idx in range(limit):
            if(len(self.unlabeled_indices) > self.unlab_sample_dim):
                seq = random.sample(self.unlabeled_indices, self.unlab_sample_dim)
            else: seq = self.unlabeled_indices
            
            logger.info(f'Index: {idx} \t Last 5 observations: {seq[-5:]}')
            
            unlabeled_subset = Subset(self.non_transformed_trainset, seq)
            
            d_labels = count_class_observation(self.classes, unlabeled_subset)
            
            self.unlab_sampled_list.append(unlabeled_subset)
            for x in seq: self.unlabeled_indices.remove(x)
            logger.info(f'Observations per class: \t{d_labels}')
        
        logger.info(f'Length of the unlabeled sampled subset list: {len(self.unlab_sampled_list)}')
        random.seed(10001) # reset the random seed
        del self.unlabeled_indices
        
        
    
    # check if all the subsets have been viewed
    def check_iterated_unlab_sampled_list(self) -> bool:
        return all(not subset.indices for subset in self.unlab_sampled_list)
        
        
        
    def run(self, epochs: int) -> Dict[str, List[float]]:
                
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        logger.info(f'----------------------- ITERATION {self.iter} / {self.al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, self.n_top_k_obs, self.iter, results)
        
        
        # start of the loop
        while not self.check_iterated_unlab_sampled_list() and self.iter < self.al_iters:
            
            # seting the indices of the subset list
            idx_list = (self.iter * self.n_top_k_obs) // self.unlab_sample_dim
            
            self.iter += 1
            
            logger.info(f'----------------------- ITERATION {self.iter} / {self.al_iters} -----------------------\n')
            
            logger.info(f' => Working with the unlabeled sampled list {idx_list}')
            logger.info(' START QUERY PROCESS\n')
            
            # run method query strategy
            idxs_new_labels, topk_idx_obs = self.query(self.unlab_sampled_list[idx_list], self.n_top_k_obs)
            
            d_labels = count_class_observation(self.classes, self.transformed_trainset, topk_idx_obs)
            logger.info(f' Number of observations per class added to the labeled set:\n {d_labels}\n')
            
            # Saving the tsne embeddings plot
            if self.method_name.split('_')[0] == 'GTG':
                # if we are performing GTG plot also the GTG predictions in the TSNE plot 
                self.save_tsne(idx_list, idxs_new_labels, d_labels, self.iter, self.gtg_result_prediction)
            
            else: self.save_tsne(idx_list, idxs_new_labels, d_labels, self.iter)

            # modify the datasets and dataloader and plot the tsne
            self.update_sets(topk_idx_obs, idx_list)

            # iter + 1
            self.train_evaluate_save(epochs, self.iter * self.n_top_k_obs, self.iter, results)
            
            
        return results