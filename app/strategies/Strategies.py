
from utils import count_class_observation, set_seeds
from train_evaluate.TrainEvaluate import TrainEvaluate

from torch.utils.data import Subset
import torch

from typing import List, Dict, Any


import logging
logger = logging.getLogger(__name__)




class Strategies(TrainEvaluate):
    def __init__(self, al_params:  Dict[str, Any], training_params: Dict[str, Any], LL: bool) -> None:
        
        self.iter = 1
        
        super().__init__(training_params, LL)
        
        self.al_iters: int = al_params['al_iters']
        self.n_top_k_obs: int = al_params['n_top_k_obs']
        self.unlab_sample_dim: int = al_params['unlab_sample_dim']
        
    
    
    def get_samp_unlab_subset(self) -> Subset:
        # set seed for reproducibility
        seed = self.dataset_id * (self.samp_iter * self.al_iters + (self.iter - 1))
        set_seeds(seed)
        
        rand_perm = torch.randperm(len(self.pool_unlabeled_indices)).tolist()
        rand_perm_unlabeled = [self.pool_unlabeled_indices[idx] for idx in rand_perm[:self.unlab_sample_dim]]
        
        logger.info(f' SEED: {seed} - Last 10 permuted indices are: {rand_perm[-10:]}')
        unlab_perm_subset = Subset(self.non_transformed_trainset, rand_perm_unlabeled)
        logger.info(f' SEED: {seed} - With dataset indices: {unlab_perm_subset.indices[-10:]}')
        
        #reset the original seed
        set_seeds()
        
        return unlab_perm_subset
        
        
        
    def run(self, epochs: int) -> Dict[str, List[float]]:
                
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        logger.info(f'----------------------- ITERATION {self.iter} / {self.al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, self.n_top_k_obs, self.iter, results)
        
        
        # start of the loop
        while self.iter < self.al_iters:

            self.iter += 1
            
            logger.info(f'----------------------- ITERATION {self.iter} / {self.al_iters} -----------------------\n')
            
            logger.info(f' => Getting the sampled unalbeled subset for the current iteration...')
            samp_unlab_subset = self.get_samp_unlab_subset()
            logger.info(' DONE\n')
            
            logger.info(' START QUERY PROCESS\n')
            
            # run method query strategy
            idxs_new_labels, topk_idx_obs = self.query(samp_unlab_subset, self.n_top_k_obs)
            
            d_labels = count_class_observation(self.classes, self.transformed_trainset, topk_idx_obs)
            logger.info(f' Number of observations per class added to the labeled set:\n {d_labels}\n')
            
            # Saving the tsne embeddings plot
            if self.method_name.split('_')[0] == 'GTG':
                # if we are performing GTG plot also the GTG predictions in the TSNE plot 
                self.save_tsne(samp_unlab_subset, idxs_new_labels, d_labels, self.iter, self.gtg_result_prediction)
            
            else: self.save_tsne(samp_unlab_subset, idxs_new_labels, d_labels, self.iter)

            # modify the datasets and dataloader and plot the tsne
            self.update_sets(topk_idx_obs, samp_unlab_subset.indices, self.unlab_sample_dim)

            # iter + 1
            self.train_evaluate_save(epochs, self.iter * self.n_top_k_obs, self.iter, results)
            
            
        return results