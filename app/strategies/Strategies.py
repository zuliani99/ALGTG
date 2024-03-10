
from train_evaluate.TrainEvaluate import TrainEvaluate

from torch.utils.data import Subset

from typing import List, Dict, Any


class Strategies(TrainEvaluate):
    def __init__(self, al_params: Dict[str, Any], LL: bool) -> None:
        super().__init__(al_params, LL)
        
    def run(self, al_iters: int, epochs: int, unlab_sample_dim: int, n_top_k_obs: int) -> Dict[str, List[float]]:
        
        self.iter = 1
        
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        print(f'----------------------- ITERATION {self.iter} / {al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, n_top_k_obs, self.iter, results)
        
        # start of the loop
        while len(self.unlabeled_indices) > 0 and self.iter < al_iters:
            self.iter += 1
            
            print(f'----------------------- ITERATION {self.iter} / {al_iters} -----------------------\n')
            
            print(f' => Sampling {unlab_sample_dim} observation from the entire unlabeled set')
            sample_unlab_subset = Subset(
                self.non_transformed_trainset,
                self.get_unlabebled_samples(unlab_sample_dim, self.iter)
            )
            print(' DONE\n')
            
            # run method query strategy
            topk_idx_obs = self.query(sample_unlab_subset, n_top_k_obs)
                    
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders(topk_idx_obs)
            print(' DONE\n')
            

            # iter + 1
            self.train_evaluate_save(epochs, self.iter * n_top_k_obs, self.iter, results)
                        
        self.remove_model_opt()
            
        return results