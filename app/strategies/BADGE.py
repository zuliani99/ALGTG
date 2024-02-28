
import torch
from torch.utils.data import DataLoader, Subset

from scipy import stats

from TrainEvaluate import TrainEvaluate
from Datasets import UniqueShuffle



class BADGE(TrainEvaluate):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.LL = LL
        

    def init_centers(self, n_top_k_obs):
        
        ind = torch.argmax([torch.norm(s, dim=2) for s in self.unlab_embeddings])
        
        mu = [self.unlab_embeddings[ind]]
        indsAll = [ind]
        centInds = [0.] * len(self.unlab_embeddings)
        cent = 0
        
        #print('#Samps\tTotal Distance')
        while len(mu) < n_top_k_obs:
            if len(mu) == 1:
                D2 = torch.cdist(self.unlab_embeddings, mu).ravel().astype(float)
            else:
                newD = torch.cdist(self.unlab_embeddings, [mu[-1]]).ravel().astype(float)
                for i in range(len(self.unlab_embeddings)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            #print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(torch.arange(len(D2).numpy()), Ddist.cpu().numpy()))
            ind = customDist.rvs(size=1)[0]
            mu.append(self.unlab_embeddings[ind])
            indsAll.append(ind)
            cent += 1
            
        return indsAll
    

    def run(self, al_iters, epochs, unlab_sample_dim, n_top_k_obs):
        
        iter = 1
        
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, n_top_k_obs, iter, results)
        
        # start of the loop
        while len(self.unlab_train_subset) > 0 and iter < al_iters:
            iter += 1
            
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
                                        
            sample_unlab_subset = Subset(
                self.non_transformed_trainset,
                self.get_unlabebled_samples(unlab_sample_dim, iter)
            )
            
            # set the entire batch size to the dimension of the sampled unlabeled set
            self.unlab_batch_size = len(sample_unlab_subset)
                        
            self.unlab_train_dl = DataLoader(
                sample_unlab_subset,
                # we have the batch size which is equal to the number of sampled observation from the unlabeled set
                batch_size=self.unlab_batch_size,
                sampler=UniqueShuffle(sample_unlab_subset),
                pin_memory=True
            )
            
            print(' => Getting the unlabeled embeddings')
            self.unlab_embeddings, _,  unlabeled_indices = self.get_embeddings(self.unlab_train_dl)
            print(' DONE\n')
                        
            print*(' => Top K extraction using init_centers')
            indices_to_convert = self.init_centers(unlabeled_indices, n_top_k_obs)
            print(' DONE\n')
                        
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders([unlabeled_indices[id].item() for id in indices_to_convert])
            print(' DONE\n')
            
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
            
        self.remove_model_opt()
            
        return results
        