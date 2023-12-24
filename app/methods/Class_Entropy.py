
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from termcolor import colored
from tqdm import tqdm
from utils import entropy, write_csv
import copy


class Class_Entropy:
    
    def __init__(self, Main_AL_class, method_params):
        self.method_name = 'Class_Entropy'
        self.Main_AL_class = Main_AL_class
        
        self.model = self.Main_AL_class.model
        
        self.params = method_params                
    
        
            
    def evaluate_unlabeled(self):

        self.model.eval()

        pbar = tqdm(self.unlab_train_dl, total = len(self.unlab_train_dl), leave=False, desc = 'TESTING ON UNLABELED')
        
        prob_dist = torch.empty(0, self.Main_AL_class.n_classes, dtype=torch.float32).to(self.Main_AL_class.device)  
        indices = torch.empty(0, dtype=torch.int8).to(self.Main_AL_class.device) 
        
        with torch.inference_mode(): # Allow inference mode
            for idxs, images, label in pbar:
                
                images, label = images.to(self.Main_AL_class.device), label.to(self.Main_AL_class.device)

                output = self.model(images)
                softmax = F.softmax(output, dim=1)
                
                prob_dist = torch.cat((prob_dist, softmax), dim = 0).to(self.Main_AL_class.device)
                indices = torch.cat((indices, idxs), dim = 0)
                
        return indices, prob_dist
    
    
    
    def get_new_dataloaders(self, overall_topk):
        #unlab_train_indices
        #lab_train_indices
        
        self.Main_AL_class.lab_train_indices.extend(overall_topk)
        self.lab_train_ds = Subset(self.Main_AL_class.train_ds, self.Main_AL_class.lab_train_indices)

        #new_idxs_unlab_train_ds = self.unlab_train_ds.indices
        for idx_to_remove in overall_topk:
            self.Main_AL_class.unlab_train_indices.remove(idx_to_remove)
        self.unlab_train_ds = Subset(self.Main_AL_class.train_ds, self.Main_AL_class.unlab_train_indices)
        
        
        print(colored(f'!!!!!!!!!!!!!!!!!!!!!!!{list(set(self.Main_AL_class.lab_train_indices) & set(self.Main_AL_class.unlab_train_indices))}!!!!!!!!!!!!!!!!!!!!!!!', 'red'))

        self.lab_train_dl = DataLoader(self.lab_train_ds, batch_size=self.Main_AL_class.batch_size, shuffle=True)
        self.Main_AL_class.obtain_normalization()
        
        
        
        
    def run(self, al_iters, epochs, n_top_k_obs):
        results = {}
        
        for n_splits in self.params['list_n_samples']:
            
            self.lab_train_dl = copy.deepcopy(self.Main_AL_class.lab_train_dl)
        
            self.lab_train_ds = copy.deepcopy(self.Main_AL_class.lab_train_ds)
            self.unlab_train_ds = copy.deepcopy(self.Main_AL_class.unlab_train_ds)
            
                    
            print(colored(f'----------------------- WORKING WITH {n_splits} UNLABELED SPLITS -----------------------\n', 'green'))
                    
            iter = 0

            results[n_splits] = { 'test_accuracy': [] }
                
            # iter = 0            
            print(colored(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n', 'blue'))
            self.Main_AL_class.reintialize_model()
            self.Main_AL_class.fit(epochs, self.lab_train_dl) # train in the labeled observations
            
            test_accuracy = self.Main_AL_class.test_AL()
                
            write_csv(
                ts_dir=self.Main_AL_class.timestamp,
                head = ['method', 'al_iter', 'n_splits', 'test_accuracy'],
                values = [self.method_name, iter, n_splits, test_accuracy]
            )
                
            results[n_splits]['test_accuracy'].append(test_accuracy)
                     
            # start of the loop   
            while len(self.unlab_train_ds) > 0 and iter < al_iters:
                print(colored(f'----------------------- ITERATION {iter + 1} / {al_iters} -----------------------\n', 'blue')) 
                
                iter_batch_size = len(self.unlab_train_ds) // n_splits
                
                self.unlab_train_dl = DataLoader(self.unlab_train_ds, batch_size=iter_batch_size, shuffle=True, num_workers=2)
                
                indices_prob, prob_dist = self.evaluate_unlabeled()
                
                tot_entr = entropy(prob_dist)
                
                overall_topk = torch.topk(tot_entr, n_top_k_obs)
                
                self.get_new_dataloaders([indices_prob[id].item() for id in overall_topk.indices])
                
                # iter + 1
                self.Main_AL_class.reintialize_model()
                self.Main_AL_class.fit(epochs, self.lab_train_dl) # train in the labeled observations
                
                test_accuracy = self.Main_AL_class.test_AL()
                
                write_csv(
                    ts_dir=self.Main_AL_class.timestamp,
                    head = ['method', 'al_iter', 'n_splits', 'test_accuracy'],
                    values = [self.method_name, iter + 1, n_splits, test_accuracy]
                )
                
                results[n_splits]['test_accuracy'].append(test_accuracy)
                        
                iter += 1
        
        return results