
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from TrainEvaluate import TrainEvaluate

from utils import entropy, save_train_val_curves, write_csv


class Class_Entropy(TrainEvaluate):
    
    def __init__(self, al_params, method_params, LL):
        super().__init__(al_params, LL)
                
        self.method_name = self.__class__.__name__
        self.params = method_params  
        self.LL = LL       
        
            
    def evaluate_unlabeled(self):

        self.model.eval()
        
        prob_dist = torch.empty(0, self.n_classes, dtype=torch.float32, device=self.device) 
        indices = torch.empty(0, dtype=torch.int8, device=self.device) 
        
        with torch.inference_mode(): # Allow inference mode
            for idxs, images, _ in self.unlab_train_dl:
                
                idxs, images = idxs.to(self.device), self.normalize(images.to(self.device))
                
                outputs, _, _, _ = self.model(images)
                    
                softmax = F.softmax(outputs, dim=1)
                
                prob_dist = torch.cat((prob_dist, softmax), dim = 0).to(self.device)
                indices = torch.cat((indices, idxs), dim = 0)
                
        return indices, prob_dist
    
        
        
    def run(self, al_iters, epochs, n_top_k_obs):
        results = {}
        
        for n_splits in self.params['list_n_samples']:           
                    
            print(f'----------------------- WORKING WITH {n_splits} UNLABELED SPLITS -----------------------\n')
                    
            iter = 0

            results[n_splits] = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
            
                
            # iter = 0            
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
            
            
            # reset the indices to the original one
            self.original_trainset.lab_train_idxs = self.lab_train_subset.indices
            self.reintialize_model()
            
            
            train_results = self.train_evaluate(epochs, self.lab_train_dl, self.method_name) # train in the labeled observations
            
            save_train_val_curves(train_results, self.timestamp, iter, self.LL)
            
            test_accuracy, test_loss, test_loss_ce, test_loss_weird = self.test()
                
            write_csv(
                ts_dir = self.timestamp,
                head = ['method', 'LL', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss', 'test_loss_ce', 'test_loss_weird'],
                values = [self.method_name, self.LL, iter, 'None', test_accuracy, test_loss, test_loss_ce, test_loss_weird]
            )
                
            results['test_accuracy'].append(test_accuracy)
            results['test_loss'].append(test_loss)
            results['test_loss_ce'].append(test_loss_ce)
            results['test_loss_weird'].append(test_loss_weird)
            
                     
            # start of the loop   
            while len(self.unlab_train_subset) > 0 and iter < al_iters:
                print(f'----------------------- ITERATION {iter + 1} / {al_iters} -----------------------\n')
                                
                self.unlab_train_dl = DataLoader(
                    self.unlab_train_subset,
                    batch_size=len(self.unlab_train_subset) // n_splits, 
                    shuffle=True, 
                    num_workers=1,
                    pin_memory=True
                )
                
                print(' => Evalueting unlabeled observations')
                indices_prob, prob_dist = self.evaluate_unlabeled()
                print(' DONE\n')
                
                tot_entr = entropy(prob_dist).to(self.device)
                overall_topk = torch.topk(tot_entr, n_top_k_obs)
                
                print(' => Modifing the Subsets and Dataloader')
                self.get_new_dataloaders([indices_prob[id].item() for id in overall_topk.indices])
                print(' DONE\n')
                
                # iter + 1
                self.reintialize_model()
                train_results = self.train_evaluate(epochs, self.lab_train_dl, self.method_name) # train in the labeled observations
                
                save_train_val_curves(train_results, self.timestamp, iter + 1, self.LL)
                
                test_accuracy, test_loss, test_loss_ce, test_loss_weird = self.test()
                
                write_csv(
                    ts_dir = self.timestamp,
                    head = ['method', 'LL', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss', 'test_loss_ce', 'test_loss_weird'],
                    values = [self.method_name, self.LL, iter, 'None', test_accuracy, test_loss, test_loss_ce, test_loss_weird]
                )
                
                results['test_accuracy'].append(test_accuracy)
                results['test_loss'].append(test_loss)
                results['test_loss_ce'].append(test_loss_ce)
                results['test_loss_weird'].append(test_loss_weird)
                        
                iter += 1
        
        return results