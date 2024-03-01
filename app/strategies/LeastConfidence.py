
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

from TrainEvaluate import TrainEvaluate


class LeastConfidence(TrainEvaluate):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.LL = LL
    

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
            
            print(' => Getting the unlabeled probebilities')
            unlabeled_indices, unlab_embeddings, _  = self.get_embeddings(self.unlab_train_dl)
            unlab_probs = F.softmax(unlab_embeddings, dim=1)
            print(' DONE\n')
            
            uncertainties = unlab_probs.max(1)[0]
            indices_to_convert = torch.topk(uncertainties, n_top_k_obs)
                                    
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders([unlabeled_indices[id].item() for id in indices_to_convert])
            print(' DONE\n')
            
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
            
        self.remove_model_opt()
            
        return results
        