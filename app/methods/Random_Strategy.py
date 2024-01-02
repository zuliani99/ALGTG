
import random
from termcolor import colored

from TrainEvaluate import TrainEvaluate
from utils import write_csv



class Random_Strategy(TrainEvaluate):
    
    def __init__(self, al_params):
        super().__init__(al_params)
        
        self.method_name = self.__class__.__name__           
                
        
        
    def sample_unlab_obs(self, n_top_k_obs):
        if(len(self.unlab_train_ds.indices) > n_top_k_obs):
            return random.sample(self.unlab_train_ds.indices, n_top_k_obs)
        else:
            return self.unlab_train_ds.indices
    

    
    def run(self, al_iters, epochs, n_top_k_obs):
        iter = 0
        results = { 'test_accuracy': [], 'test_loss': [] }
        
        # iter = 0
        print(colored(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n', 'blue'))
            
        self.reintialize_model()
        self.fit(epochs, self.lab_train_dl, self.method_name)
            
        test_accuracy, test_loss = self.test_AL()
        
        write_csv(
            ts_dir=self.timestamp,
            head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
            values = [self.method_name, iter, 'None', test_accuracy, test_loss]
        )
        
        results['test_accuracy'].append(test_accuracy)
        results['test_loss'].append(test_loss)
        
        # start of the loop
        while len(self.unlab_train_ds) > 0 and iter < al_iters:
            print(colored(f'----------------------- ITERATION {iter + 1} / {al_iters} -----------------------\n', 'blue'))
                            
            #get random indices to move in the labeled datasets
            topk_idx_obs = self.sample_unlab_obs(n_top_k_obs)
                        
            # modify the datasets and dataloader
            self.get_new_dataloaders(topk_idx_obs)
            
            
            # iter + 1
            self.reintialize_model()
            self.fit(epochs, self.lab_train_dl, self.method_name)
            
            test_accuracy, test_loss = self.test_AL()
            
            write_csv(
                ts_dir=self.timestamp,
                head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                values = [self.method_name, iter + 1, 'None', test_accuracy, test_loss]
            )

            results['test_accuracy'].append(test_accuracy)
            results['test_loss'].append(test_loss)
            
            iter += 1
            
        return results