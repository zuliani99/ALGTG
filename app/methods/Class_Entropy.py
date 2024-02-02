
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from TrainEvaluate import TrainEvaluate

from utils import entropy, save_train_val_curves, write_csv

from tqdm import tqdm


class Class_Entropy(TrainEvaluate):
    
    def __init__(self, al_params, method_params):
        super().__init__(al_params)
                
        self.method_name = self.__class__.__name__
        self.params = method_params                
    
        
            
    def evaluate_unlabeled(self):

        self.model.eval()

        #pbar = tqdm(self.unlab_train_dl, total = len(self.unlab_train_dl), leave=False, desc = 'TESTING ON UNLABELED')
        
        prob_dist = torch.empty(0, self.n_classes, dtype=torch.float32).to(self.device)  
        indices = torch.empty(0, dtype=torch.int8).to(self.device) 
        
        with torch.inference_mode(): # Allow inference mode
            for idxs, images, labels in self.unlab_train_dl:
                
                idxs, images, labels = idxs.to(self.device), self.normalize(images.to(self.device)), labels.to(self.device)
                
                
                #if self.model.__class__.__name__ == 'ResNet_Weird':
                outputs, _, _, _ = self.model(images)
                #else:
                    #outputs = self.model(images)
                    
                    
                softmax = F.softmax(outputs, dim=1)
                
                prob_dist = torch.cat((prob_dist, softmax), dim = 0).to(self.device)
                indices = torch.cat((indices, idxs), dim = 0)
                
        return indices, prob_dist
    
        
        
    def run(self, al_iters, epochs, n_top_k_obs):
        results = {}
        
        for n_splits in self.params['list_n_samples']:           
                    
            print(f'----------------------- WORKING WITH {n_splits} UNLABELED SPLITS -----------------------\n')
                    
            iter = 0

            results[n_splits] = { 'test_accuracy': [], 'test_loss': [] }
                
            # iter = 0            
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
            
            
            # reset the indices to the original one
            self.original_trainset.lab_train_idxs = self.lab_train_subset.indices
            self.reintialize_model()
            
            
            train_results = self.fit(epochs, self.lab_train_dl, self.method_name) # train in the labeled observations
            
            save_train_val_curves(train_results, self.timestamp, iter)
            
            test_accuracy, test_loss = self.test_AL()
                
            write_csv(
                ts_dir=self.timestamp,
                head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                values = [self.method_name, iter, n_splits, test_accuracy, test_loss]
            )
                
            results[n_splits]['test_accuracy'].append(test_accuracy)
            results[n_splits]['test_loss'].append(test_loss)
            
                     
            # start of the loop   
            while len(self.unlab_train_subset) > 0 and iter < al_iters:
                print(f'----------------------- ITERATION {iter + 1} / {al_iters} -----------------------\n')
                
                iter_batch_size = len(self.unlab_train_subset) // n_splits
                
                self.unlab_train_dl = DataLoader(self.unlab_train_subset, batch_size=iter_batch_size, shuffle=True, num_workers=1, pin_memory=True)
                
                indices_prob, prob_dist = self.evaluate_unlabeled()
                
                tot_entr = entropy(prob_dist)
                
                overall_topk = torch.topk(tot_entr, n_top_k_obs)
                
                self.get_new_dataloaders([indices_prob[id].item() for id in overall_topk.indices])
                
                # iter + 1
                self.reintialize_model()
                train_results = self.fit(epochs, self.lab_train_dl, self.method_name) # train in the labeled observations
                
                save_train_val_curves(train_results, self.timestamp, iter + 1)
                
                test_accuracy, test_loss = self.test_AL()
                
                write_csv(
                    ts_dir=self.timestamp,
                    head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                    values = [self.method_name, iter + 1, n_splits, test_accuracy, test_loss]
                )
                
                results[n_splits]['test_accuracy'].append(test_accuracy)
                results[n_splits]['test_loss'].append(test_loss)
                        
                iter += 1
        
        return results