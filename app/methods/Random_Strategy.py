
import random
from termcolor import colored
import copy

from torch.utils.data import DataLoader, Subset

from utils import write_csv



class Random_Strategy():
    def __init__(self, Main_AL_class):
        self.method_name = 'Random'
        self.Main_AL_class = Main_AL_class
        
        self.lab_train_ds = copy.deepcopy(self.Main_AL_class.lab_train_ds)
        self.unlab_train_ds = copy.deepcopy(self.Main_AL_class.unlab_train_ds)
        
        self.lab_train_dl = copy.deepcopy(self.Main_AL_class.lab_train_dl)
                
        
        
    def sample_unlab_obs(self, n_top_k_obs):
        if(len(self.unlab_train_ds.indices) > n_top_k_obs):
            return random.sample(self.unlab_train_ds.indices, n_top_k_obs)
        else:
            return self.unlab_train_ds.indices
    
    
    def get_new_dataloaders(self, overall_topk):
        
        lab_train_indices = self.lab_train_ds.indices
        
        lab_train_indices.extend(overall_topk)
        self.lab_train_ds = Subset(self.Main_AL_class.train_ds, lab_train_indices)

        unlab_train_indices = self.unlab_train_ds.indices
        for idx_to_remove in overall_topk:
            unlab_train_indices.remove(idx_to_remove)
        self.unlab_train_ds = Subset(self.Main_AL_class.train_ds, unlab_train_indices)      
        
        print(colored(f'!!!!!!!!!!!!!!!!!!!!!!!{list(set(self.unlab_train_ds.indices) & set(self.lab_train_ds.indices))}!!!!!!!!!!!!!!!!!!!!!!!', 'red'))

        self.lab_train_dl = DataLoader(self.lab_train_ds, batch_size=self.Main_AL_class.batch_size, shuffle=True) # False
        self.Main_AL_class.obtain_normalization()



    
    def run(self, al_iters, epochs, n_top_k_obs):
        iter = 0
        results = { 'test_accuracy': [] }
        
        # iter = 0
        print(colored(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n', 'blue'))
            
        self.Main_AL_class.reintialize_model()
        self.Main_AL_class.fit(epochs, self.lab_train_dl, self.method_name)
            
        test_accuracy = self.Main_AL_class.test_AL()
        
        write_csv(
            ts_dir=self.Main_AL_class.timestamp,
            head = ['method', 'al_iter', 'n_splits', 'test_accuracy'],
            values = [self.method_name, iter, 'None', test_accuracy]
        )
        
        results['test_accuracy'].append(test_accuracy)
        
        # start of the loop
        while len(self.unlab_train_ds) > 0 and iter < al_iters:
            print(colored(f'----------------------- ITERATION {iter + 1} / {al_iters} -----------------------\n', 'blue'))
                            
            #get random indices to move in the labeled datasets
            topk_idx_obs = self.sample_unlab_obs(n_top_k_obs)
                        
            # modify the datasets and dataloader
            self.get_new_dataloaders(topk_idx_obs)
            
            
            # iter + 1
            self.Main_AL_class.reintialize_model()
            self.Main_AL_class.fit(epochs, self.lab_train_dl, self.method_name)
            
            test_accuracy = self.Main_AL_class.test_AL()
            
            write_csv(
                ts_dir=self.Main_AL_class.timestamp,
                head = ['method', 'al_iter', 'n_splits', 'test_accuracy'],
                values = [self.method_name, iter + 1, 'None', test_accuracy]
            )

            results['test_accuracy'].append(test_accuracy)
            
            iter += 1
            
        return results