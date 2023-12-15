
import random
import numpy as np
from tqdm import tqdm
from termcolor import colored

from torch.utils.data import DataLoader

from utils import write_csv
from cifar10 import CIFAR10



class Random_Strategy():
    def __init__(self, Main_AL_class):
        self.method_name = 'random'
        self.Main_AL_class = Main_AL_class
        
        self.lab_train_ds = self.Main_AL_class.lab_train_ds
        self.unlab_train_ds = self.Main_AL_class.unlab_train_ds
        
        self.lab_train_dl = self.Main_AL_class.lab_train_dl
                
        
        
    def sample_unlab_obs(self, n_top_k_obs):
        return random.sample(range(len(self.unlab_train_ds)), n_top_k_obs)
    
    
    
    def get_new_dataloaders(self, topk_idx_obs):
        new_lab_train_ds = np.array([
            np.array([
                image if isinstance(image, np.ndarray) else image.numpy(), label
            ], dtype=object) for image, label in tqdm(self.lab_train_ds, total=len(self.lab_train_ds), leave=False, desc='Copying lab_train_ds')], dtype=object)
        
        
        new_unlab_train_ds = np.array([
            np.array([
                image if isinstance(image, np.ndarray) else image.numpy(), label
            ], dtype=object) for image, label in tqdm(self.unlab_train_ds, total=len(self.unlab_train_ds), leave=False, desc='Copying unlab_train_ds')], dtype=object)
        


        for idx_to_move in tqdm(topk_idx_obs, total=len(topk_idx_obs), leave=False, desc='Modifing the Unlabeled Dataset'):
            new_lab_train_ds = np.vstack((new_lab_train_ds, np.expand_dims(
                np.array(new_unlab_train_ds[idx_to_move], dtype=object)
            , axis=0)))
            
            new_unlab_train_ds[idx_to_move] = np.array([np.nan, np.nan], dtype=object) # set a [np.nan np.nan] the row and the get all the row not equal to [np.nan, np.nan]       
            
        
        
        self.lab_train_ds = CIFAR10(None, new_lab_train_ds)
        self.unlab_train_ds = CIFAR10(None,
                                      new_unlab_train_ds[np.array([not np.isnan(row[1])
                                                                   for row in tqdm(new_unlab_train_ds, total=len(new_unlab_train_ds), desc='Obtaining the unmarked observation from the Unlabeled Dataset', leave=False)])])

        self.lab_train_dl = DataLoader(self.lab_train_ds, batch_size=self.Main_AL_class.batch_size, shuffle=True)
        
    
    
    def run(self, al_iters, epochs, n_top_k_obs):
        iter = 0
        results = { 'test_loss': [], 'test_accuracy': [] }
        
        # iter = 0
        print(colored(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n', 'blue'))
            
        self.Main_AL_class.reintialize_model()
        self.Main_AL_class.fit(epochs, self.lab_train_dl)
            
        test_loss, test_accuracy = self.Main_AL_class.test_AL()
        
        write_csv(
            ts_dir=self.Main_AL_class.timestamp,
            head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
            values = [self.method_name, iter, 'None', test_accuracy, test_loss]
        )

        
        results['test_loss'].append(test_loss)
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
            self.Main_AL_class.fit(epochs, self.lab_train_dl)
            
            test_loss, test_accuracy = self.Main_AL_class.test_AL()
            
            write_csv(
                ts_dir=self.Main_AL_class.timestamp,
                head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                values = [self.method_name, iter + 1, 'None', test_accuracy, test_loss]
            )

            results['test_loss'].append(test_loss)
            results['test_accuracy'].append(test_accuracy)
            
            
            iter += 1
            
            
        return results