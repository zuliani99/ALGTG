
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
        
        
        
    def run(self, al_iters, epochs, unlab_sample_dim, n_top_k_obs):
                            
        iter = 1
            
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
            
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
            
        # iter = 0
        self.train_evaluate_save(epochs, n_top_k_obs, iter, results)
            
        # start of the loop   
        while len(self.unlabeled_indices) > 0 and iter < al_iters:
            iter += 1
                
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
                            
            # get random unlabeled sampple
            sample_unlab = self.get_unlabebled_samples(unlab_sample_dim, iter)
                                
            self.unlab_train_dl = DataLoader(
                Subset(self.non_transformed_trainset, sample_unlab),
                batch_size=self.batch_size, shuffle=True, pin_memory=True
            )
                
            print(' => Evalueting unlabeled observations')
            indices_prob, embeddings, _ = self.get_embeddings(self.unlab_train_dl)
            prob_dist = F.softmax(embeddings, dim=1)
            print(' DONE\n')
                
            tot_entr = entropy(prob_dist).to(self.device)
            overall_topk = torch.topk(tot_entr, n_top_k_obs)
                
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders([indices_prob[id].item() for id in overall_topk.indices])
            print(' DONE\n')
                            
            self.remove_idxs_probs(indices_prob, prob_dist)
                            
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
            
        return results