
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from TrainEvaluate import TrainEvaluate

from utils import entropy


class Entropy(TrainEvaluate):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
                
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.LL = LL
        
            
            
    def evaluate_unlabeled(self):

        self.model.eval()
        
        prob_dist = torch.empty(0, self.n_classes, dtype=torch.float32, device=self.device) 
        indices = torch.empty(0, dtype=torch.int8, device=self.device) 
        
        with torch.inference_mode(): # Allow inference mode
            for idxs, images, _ in self.unlab_train_dl:
                
                idxs, images = idxs.to(self.device), images.to(self.device)
                
                outputs, _, _, _ = self.model(images)
                    
                softmax = F.softmax(outputs, dim=1)
                
                prob_dist = torch.cat((prob_dist, softmax), dim = 0).to(self.device)
                indices = torch.cat((indices, idxs), dim = 0)
                
        return indices, prob_dist
    
    
    
    def remove_idxs_probs(self, indices_prob, prob_dist):
        del indices_prob
        del prob_dist
        torch.cuda.empty_cache()
        
        
        
    def run(self, al_iters, epochs, unlab_sample_dim, n_top_k_obs):
                            
        iter = 1
            
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
            
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
            
        # iter = 0
        self.train_evaluate_save(epochs, n_top_k_obs, iter, results)
            
        # start of the loop   
        while len(self.unlab_train_subset) > 0 and iter < al_iters:
            iter += 1
                
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
                            
            # get random unlabeled sampple
            sample_unlab = self.get_unlabebled_samples(unlab_sample_dim, iter)
                                
            self.unlab_train_dl = DataLoader(
                Subset(self.non_transformed_trainset, sample_unlab),
                batch_size=self.batch_size, shuffle=True, pin_memory=True
            )
                
            print(' => Evalueting unlabeled observations')
            indices_prob, prob_dist = self.evaluate_unlabeled()
            print(' DONE\n')
                
            tot_entr = entropy(prob_dist).to(self.device)
            overall_topk = torch.topk(tot_entr, n_top_k_obs)
                
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders([indices_prob[id].item() for id in overall_topk.indices])
            print(' DONE\n')
                            
            self.remove_idxs_probs(indices_prob, prob_dist)
                            
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
            
        self.remove_model_opt()
                    
        return results