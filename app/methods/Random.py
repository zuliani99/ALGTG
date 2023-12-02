

from utils import write_csv
import random
import numpy as np
from tqdm import tqdm
from cifar10 import CIFAR10
from torch.utils.data import DataLoader



class Random_Strategy():
    def __init__(self, Main_AL_class):#, random_param):
        self.method_name = 'random'
        self.Main_AL_class = Main_AL_class
        
        #self.random_param = random_param
        
        self.lab_train_dl = self.Main_AL_class.lab_train_dl
        self.unlab_train_dl = self.Main_AL_class.unlab_train_dl
        
        self.lab_train_ds = self.Main_AL_class.lab_train_ds
        self.unlab_train_ds = self.Main_AL_class.unlab_train_ds
        
        
        
    def sample_unlab_obs(self, n_top_k_obs):
        return random.sample(range(len(self.unlab_train_ds)), n_top_k_obs)   
    
    
    
    def get_new_dataloaders(self, topk_idx_obs):
        new_lab_train_ds = np.array([
            np.array([
                self.lab_train_ds[i][0] if isinstance(self.lab_train_ds[i][0], np.ndarray) else self.lab_train_ds[i][0].numpy(),
                self.lab_train_ds[i][1]
            ], dtype=object) for i in tqdm(range(len(self.lab_train_ds)), leave=False)], dtype=object)
        
        #new_lab_train_ds = torch.tensor([self.lab_train_ds[i][::1]] for i in tqdm(range(len(self.lab_train_ds))))
        

        new_unlab_train_ds = np.array([
            np.array([
                self.unlab_train_ds[i][0] if isinstance(self.unlab_train_ds[i][0], np.ndarray) else self.unlab_train_ds[i][0].numpy(),
                self.unlab_train_ds[i][1]
            ], dtype=object) for i in tqdm(range(len(self.unlab_train_ds)), leave=False)], dtype=object)
        
        #new_unlab_train_ds = torch.tensor([self.unlab_train_ds[i][::1]] for i in tqdm(range(len(self.unlab_train_ds))))



        #for list_index, topk_index_value in topk_idx_obs:
        for idx_to_move in topk_idx_obs:
            new_lab_train_ds = np.vstack((new_lab_train_ds, np.expand_dims(
                np.array([
                    self.unlab_train_ds[idx_to_move][0] if isinstance(self.unlab_samp_list[idx_to_move][0], np.ndarray)
                        else self.unlab_train_ds[idx_to_move][0].numpy(),
                    self.unlab_train_ds[idx_to_move][1]
                ], dtype=object)
            , axis=0)))
            
            #new_lab_train_ds = torch.cat((new_lab_train_ds, torch.tensor([self.unlab_samp_list[list_index][0][topk_index_value][::1]])), dim = 0)
            
        #for list_index, topk_index_value in overall_topk:
        for idx_to_move in topk_idx_obs:
            new_unlab_train_ds = np.delete(new_unlab_train_ds, idx_to_move, axis = 0)
            
            #new_unlab_train_ds = torch.cat((new_unlab_train_ds[ : ((list_index * n_samples) + topk_index_value)],
            #                                new_unlab_train_ds[((list_index * n_samples) + topk_index_value + 1) : ]))
            


        self.lab_train_ds = CIFAR10(None, new_lab_train_ds)
        self.unlab_train_ds = CIFAR10(None, new_unlab_train_ds)

        self.lab_train_dl = DataLoader(self.lab_train_ds, batch_size=self.Main_AL_class.batch_size, shuffle=True)
        self.unlab_train_dl = DataLoader(self.unlab_train_ds, batch_size=self.Main_AL_class.batch_size, shuffle=True)
        
    
    
    def run(self, al_iters, epochs, n_top_k_obs):
        iter = 0
        results = { 'test_loss': [], 'test_accuracy': [] }
        
        
        while len(self.unlab_train_ds) > 0 and iter < al_iters:
            print(f'----------------------- ITERATION {iter + 1} / {al_iters} -----------------------\n')
            
            
            self.Main_AL_class.reintialize_model()
            self.Main_AL_class.fit(epochs, self.lab_train_dl)
            
            test_loss, test_accuracy = self.test_AL_GTG()
            
            
            write_csv(
                filename = 'RANDOM_test_res.csv',
                head = ['method', 'al_iter', 'test_accuracy', 'test_loss'],
                values = [self.method_name, iter, test_accuracy, test_loss]
            )
                
            
            self.labeled_embeddings = self.Main_AL_class.get_embeddings(self.lab_train_dl)
            self.unlabeled_embeddings = self.Main_AL_class.get_embeddings(self.unlab_train_dl)
                        
                        
            #get random indices to move in the labeled datasets
            topk_idx_obs = self.sample_unlab_obs(n_top_k_obs)
                        
            # modify the datasets and dataloader
            self.get_new_dataloaders(topk_idx_obs)
                        
            
            
                        
            results['test_loss'].append(test_loss)
            results['test_accuracy'].append(test_accuracy)
            
            iter += 1
            
            
        return results