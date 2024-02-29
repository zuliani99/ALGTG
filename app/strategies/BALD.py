
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from TrainEvaluate import TrainEvaluate
from Datasets import UniqueShuffle

from utils import entropy


class BALD(TrainEvaluate):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.LL = LL
        
        
    def evaluate_unlabeled_train(self, n_drop=5):

        self.model.train()
        
        prob_dist_drop = torch.zeros((n_drop, len(self.unlab_train_dl.dataset), self.n_classes), dtype=torch.float32, device=self.device) 
        indices = torch.empty(0, dtype=torch.int8, device=self.device) 
        
        for drop in range(n_drop):
            with torch.inference_mode(): # Allow inference mode
                for idx_dl, (idxs, images, _) in enumerate(self.unlab_train_dl):
                    
                    idxs, images = idxs.to(self.device), images.to(self.device)
                    
                    outputs, _, _, _ = self.model(images)
                                                         
                    prob_dist_drop[drop][idx_dl * idxs.shape[0] : (idx_dl + 1) * idxs.shape[0]] += F.softmax(outputs, dim=1)

                    if(drop == 0): indices = torch.cat((indices, idxs), dim = 0)
                    
        return indices, prob_dist_drop 
        
    

    def disagreement_dropout(self):
        indices, prob_dist_drop = self.evaluate_unlabeled_train()
        
        mean_pb = torch.mean(prob_dist_drop, dim=0)
        
        entropy1 = entropy(mean_pb)
        entropy2 = torch.mean(entropy(prob_dist_drop, dim=2), dim=0)
        
        del prob_dist_drop
        torch.cuda.empty_cache()
        
        return indices, entropy2 - entropy1
        
    
    
    def run(self, al_iters, epochs, unlab_sample_dim, n_top_k_obs):
        
        iter = 1
        
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, n_top_k_obs, iter, results)
        
        # start of the loop
        while len(self.unlabeled_indices) > 0 and iter < al_iters:
            iter += 1
            
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')

            sample_unlab_subset = Subset(
                self.non_transformed_trainset,
                self.get_unlabebled_samples(unlab_sample_dim, iter)
            )
                        
            self.unlab_train_dl = DataLoader(
                sample_unlab_subset, batch_size=self.batch_size,
                shuffle=True, pin_memory=True
            )
            
            print(' => Performing the disagreement dropout')
            indices, res_entropy = self.disagreement_dropout()
            print(' DONE\n')
            
            overall_topk = torch.topk(res_entropy, n_top_k_obs)
            
            #print(overall_topk)
                        
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders([indices[id].item() for id in overall_topk.indices])
            print(' DONE\n')
            
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
            
        self.remove_model_opt()
            
        return results
        