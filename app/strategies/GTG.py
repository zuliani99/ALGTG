
from TrainEvaluate import TrainEvaluate
from utils import entropy, plot_story_tensor
from Datasets import UniqueShuffle

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

import copy


class GTG(TrainEvaluate):
    
    def __init__(self, al_params, our_methods_params, LL, A_function, zero_diag):
        super().__init__(al_params, LL)
        
        self.get_A_fn = {
            'cos_sim': self.get_A_cos_sim,
            'corr': self.get_A_corr,
        }
        self.A_function = A_function
        self.zero_diag = zero_diag
        
        str_diag = '0diag' if self.zero_diag else '1diag'
                
        self.method_name = f'{self.__class__.__name__}_{self.A_function}_{str_diag}_LL' if LL else f'{self.__class__.__name__}_{self.A_function}_{str_diag}'
        self.params = our_methods_params 
        self.LL = LL
        
        
    # select the relative choosen affinity matrix method
    def get_A(self, samp_unlab_embeddings): self.get_A_fn[self.A_function](samp_unlab_embeddings)



    def clear_memory(self):
        del self.X
        del self.A
        torch.cuda.empty_cache()



    def get_A_cos_sim(self, samp_unlab_embeddings):
                        
        normalized_embedding = F.normalize(
            torch.cat((self.labeled_embeddings, samp_unlab_embeddings)).to(self.device)
        , dim=-1).to(self.device)
        
        self.A = F.relu(torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device))
        
        if self.zero_diag: self.A.fill_diagonal_(0.)
        else: self.A.fill_diagonal_(1.)
        
        del normalized_embedding
        
        
        
        
    def get_A_corr(self, samp_unlab_embeddings):
        self.A = F.relu(torch.corrcoef(torch.cat((self.labeled_embeddings, samp_unlab_embeddings)).to(self.device)))
        if self.zero_diag: self.A.fill_diagonal_(0.)




    def get_X(self, target_lab_obs, len_samp_unlab_embeds):
        
        self.X = torch.zeros(
            (len(self.lab_train_subset) + len_samp_unlab_embeds, self.n_classes),
            dtype=torch.float32,
            device=self.device
        )

        for idx, label in enumerate(target_lab_obs): self.X[idx][label] = 1.
        
        for idx in range(len(self.lab_train_subset), len(self.lab_train_subset) + len_samp_unlab_embeds):
            for label in range(self.n_classes): self.X[idx][label] = 1. / self.n_classes
        
        


    def gtg(self, indices):
        err = float('Inf')
        i = 0
        old_rowsum_X = 0
        
        while err > self.params['gtg_tol'] and i < self.params['gtg_max_iter']:
            X_old = copy.deepcopy(self.X)
            
            self.X *= torch.mm(self.A, self.X)
            
            
            rowsum_X = torch.sum(self.X).item()
            if rowsum_X < old_rowsum_X: # it has to be increasing or at least equal
                raise Exception('Sum of the denominator vectro lower than the previous step')
            old_rowsum_X = rowsum_X


            self.X /= torch.sum(self.X, dim=1, keepdim=True)            
        
            iter_entropy = entropy(self.X).to(self.device) # both labeled and sample unlabeled
            # I have to map only the sample_unlabeled to the correct position

            # I iterate only the sampled unlabeled one                
            for idx, unlab_ent_val in zip(indices, iter_entropy[len(self.lab_train_subset):]):
                
                self.entropy_history[idx][i] = unlab_ent_val
                
                # the last step I do not save otherwise i obtain an invald access
                if (i != self.params['gtg_max_iter'] - 1): self.entropy_pairwise_der[idx][i] = unlab_ent_val

                # subtract the previous with the new entropy except for the first iteration
                if (i != 0): self.entropy_pairwise_der[idx][i - 1] = (self.entropy_pairwise_der[idx][i - 1] - unlab_ent_val)

            
            err = torch.norm(self.X - X_old)
            i += 1




    def run(self, al_iters, epochs, unlab_sample_dim, n_top_k_obs):

        iter = 1            

        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
                
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
        
        # iter = 1
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
                
            print(' => Getting the labeled and unlabeled embeddings')
            self.labeled_embeddings, target_lab_obs, _ = self.get_embeddings(self.lab_train_dl)
            self.unlab_embeddings, _, _ = self.get_embeddings(self.unlab_train_dl)
            print(' DONE\n')
            
            
            # at each AL round I reinitialize the entropy_pairwise_der since I have to decide at each step what observations I want to move
            self.entropy_pairwise_der = torch.zeros(
                (len(self.transformed_trainset), self.params['gtg_max_iter'] - 1), device=self.device
            )
            # I aslo save the entropy history in order to be able to plot it
            self.entropy_history = torch.zeros(
                (len(self.transformed_trainset), self.params['gtg_max_iter']), device=self.device
            )
            

            # for each split
            # split_idx -> indices of the split
            # indices -> are the list of indices for the given batch which ARE CONSISTENT SINCE ARE REFERRED TO THE INDEX OF THE ORIGINAL DATASET
            for split_idx, (indices, _, _) in enumerate(self.unlab_train_dl):
                print(f' => Running GTG for split {split_idx}')
                self.get_A(self.unlab_embeddings[split_idx * self.unlab_batch_size : (split_idx + 1) * self.unlab_batch_size])
                self.get_X(target_lab_obs, indices.shape[0])
                self.gtg(indices)
                self.clear_memory()                
                print(' DONE\n')
                
                
            # absolute value of the derivate to remove the oscilaltions -> observaion that have oscillation smeans that are difficult too
            self.entropy_pairwise_der = torch.abs(self.entropy_pairwise_der)


            # plot entropy history and derivatives
            plot_story_tensor(self.entropy_history, f'./app/gtg_entropy/history/{self.method_name}_{iter}.png', iter, self.params['gtg_max_iter'])
            plot_story_tensor(self.entropy_pairwise_der, f'./app/gtg_entropy/derivatives/{self.method_name}_{iter}.png', iter, self.params['gtg_max_iter'] - 1)
            
            
            # getting the last column that have at least one element with entropy greater than 1e-15
            for col_index in range(self.entropy_pairwise_der.size(1) - 1, -1, -1):
                column = self.entropy_pairwise_der[:, col_index]
                if torch.any(column > 1e-15): break
            
            
            # set the weights to increasing value until col_index
            weights = torch.zeros(self.params['gtg_max_iter'] - 1, dtype=torch.float32, device=self.device) 
            weights[:col_index] = torch.flip(torch.linspace(1, 0.1, col_index), [0]).to(self.device)
            
            # weighted average
            overall_topk = torch.topk(torch.mean(self.entropy_pairwise_der * weights, dim = 1), n_top_k_obs)
            
            
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders(overall_topk.indices.tolist())
            print(' DONE\n')
                
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
            
        self.remove_model_opt()
                                            
        return results