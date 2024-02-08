
import random

from TrainEvaluate import TrainEvaluate
from utils import save_train_val_curves, write_csv



class Random_Strategy(TrainEvaluate):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = self.__class__.__name__
        self.LL = LL
        
        
    def sample_unlab_obs(self, n_top_k_obs):
        if(len(self.unlab_train_subset.indices) > n_top_k_obs):
            return random.sample(self.unlab_train_subset.indices, n_top_k_obs)
        else:
            return self.unlab_train_subset.indices     
                

    
    def run(self, al_iters, epochs, n_top_k_obs):
        iter = 0
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        # iter = 0
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
        
        
        # reset the indices to the original one
        self.original_trainset.lab_train_idxs = self.lab_train_subset.indices
        self.reintialize_model()
        
        
        train_results = self.train_evaluate(epochs, self.lab_train_dl, self.method_name)
        
        save_train_val_curves(train_results, self.timestamp, iter, self.LL)
            
        test_accuracy, test_loss, test_loss_ce, test_loss_weird = self.test()
        
        write_csv(
            ts_dir = self.timestamp,
            head = ['method', 'LL', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
            values = [self.method_name, self.LL, iter, 'None', test_accuracy, test_loss]
        )
        
        results['test_accuracy'].append(test_accuracy)
        results['test_loss'].append(test_loss)
        results['test_loss_ce'].append(test_loss_ce)
        results['test_loss_weird'].append(test_loss_weird)
        
        # start of the loop
        while len(self.unlab_train_subset) > 0 and iter < al_iters:
            print(f'----------------------- ITERATION {iter + 1} / {al_iters} -----------------------\n')
                            
            #get random indices to move in the labeled datasets
            print(' => Sampling random unlabeled observations')
            topk_idx_obs = self.sample_unlab_obs(n_top_k_obs)
            print(' DONE\n')
            
                        
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders(topk_idx_obs)
            print(' DONE\n')
            
            
            # iter + 1
            self.reintialize_model()
            train_results = self.train_evaluate(epochs, self.lab_train_dl, self.method_name)
            
            save_train_val_curves(train_results, self.timestamp, iter + 1, self.LL)
            
            test_accuracy, test_loss, test_loss_ce, test_loss_weird = self.test()
            
            write_csv(
                ts_dir = self.timestamp,
                head = ['method', 'LL', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                values = [self.method_name, self.LL, iter, 'None', test_accuracy, test_loss]
            )

            results['test_accuracy'].append(test_accuracy)
            results['test_loss'].append(test_loss)
            results['test_loss_ce'].append(test_loss_ce)
            results['test_loss_weird'].append(test_loss_weird)
            
            iter += 1
            
        return results
        
        