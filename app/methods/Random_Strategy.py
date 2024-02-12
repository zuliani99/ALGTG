
import random

from TrainEvaluate import TrainEvaluate


class Random_Strategy(TrainEvaluate):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.LL = LL
        
        
        
    def sample_unlab_obs(self, n_top_k_obs):
        if(len(self.unlab_train_subset.indices) > n_top_k_obs):
            return random.sample(self.unlab_train_subset.indices, n_top_k_obs)
        else:
            return self.unlab_train_subset.indices
                       


    def run(self, al_iters, epochs, n_top_k_obs):
        
        iter = 0
        
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
        
        # iter = 0
        self.train_evaluate_save(epochs, n_top_k_obs, None, results)
        
        # start of the loop
        while len(self.unlab_train_subset) > 0 and iter < al_iters:
            iter += 1
            
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
                            
            #get random indices to move in the labeled datasets
            print(' => Sampling random unlabeled observations')
            topk_idx_obs = self.sample_unlab_obs(n_top_k_obs)
            print(' DONE\n')
                        
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders(topk_idx_obs)
            print(' DONE\n')
            
            # iter + 1
            self.train_evaluate_save(epochs, (iter + 1) * n_top_k_obs, None, results)
            
        return results
        