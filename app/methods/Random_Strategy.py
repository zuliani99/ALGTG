
import random

from TrainEvaluate import TrainEvaluate


class Random_Strategy(TrainEvaluate):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.LL = LL
        
        
        
    def sample_unlab_obs(self, unlab_sample_dim, n_top_k_obs, iter): 
        sample_unlab = self.get_unlabebled_samples(unlab_sample_dim, iter)
        
        if(len(sample_unlab) > n_top_k_obs):           
            return random.sample(sample_unlab, n_top_k_obs)
        else: 
            return sample_unlab                      


    def run(self, al_iters, epochs, unlab_sample_dim, n_top_k_obs):
        
        iter = 1
        
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, n_top_k_obs, iter, results)
        
        # start of the loop
        while len(self.unlab_train_subset) > 0 and iter < al_iters:
            iter += 1
            
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
                                        
            # get random indices to move in the labeled datasets
            print(' => Sampling random unlabeled observations')
            topk_idx_obs = self.sample_unlab_obs(unlab_sample_dim, n_top_k_obs, iter)
            print(' DONE\n')
                        
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders(topk_idx_obs)
            print(' DONE\n')
            
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
            
        return results
        