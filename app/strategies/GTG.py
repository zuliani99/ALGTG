
from strategies.Strategies import Strategies
from utils import entropy, plot_history, plot_derivatives

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import gpytorch
#from scipy.integrate import trapz
#import numpy as np

import copy


class GTG(Strategies):
    
    def __init__(self, al_params, our_methods_params, LL, A_function, zero_diag):
        super().__init__(al_params, LL)
        
        self.get_A_fn = {
            'cos_sim': self.get_A_cos_sim,
            'corr': self.get_A_corr,
            'rbfk': self.get_A_rbfk,
        }
        self.A_function = A_function
        self.zero_diag = zero_diag
        
        str_diag = '0diag' if self.zero_diag else '1diag'
                
        self.method_name = f'{self.__class__.__name__}_{self.A_function}_{str_diag}_LL' if LL else f'{self.__class__.__name__}_{self.A_function}_{str_diag}'
        self.params = our_methods_params 
        
        
    # select the relative choosen affinity matrix method
    def get_A(self, samp_unlab_embeddings): self.get_A_fn[self.A_function](samp_unlab_embeddings)


    def clear_memory(self):
        del self.X
        del self.A
        torch.cuda.empty_cache()


    # correct
    def get_A_cos_sim(self, samp_unlab_embeddings):
                        
        normalized_embedding = F.normalize(
            torch.cat((self.lab_embedds_dict['embedds'], samp_unlab_embeddings)).to(self.device)
        , dim=-1).to(self.device)
        
        self.A = F.relu(torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device))
        
        if self.zero_diag: self.A.fill_diagonal_(0.)
        else: self.A.fill_diagonal_(1.)
        
        del normalized_embedding
        
        
    # correct
    def get_A_corr(self, samp_unlab_embeddings):
        self.A = F.relu(torch.corrcoef(torch.cat((self.lab_embedds_dict['embedds'], samp_unlab_embeddings)).to(self.device)))
        if self.zero_diag: self.A.fill_diagonal_(0.)


    # correct
    #https://docs.gpytorch.ai/en/stable/kernels.html#rbfkernel
    def get_A_rbfk(self, samp_unlab_embeddings):
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(self.device)
        self.A = covar_module(torch.cat((self.lab_embedds_dict['embedds'], samp_unlab_embeddings)).to(self.device)).evaluate().detach()
        if self.zero_diag: self.A.fill_diagonal_(0.)
        

    # correct
    def get_X(self, len_samp_unlab_embeds):
        
        self.X = torch.zeros(
            (len(self.labeled_indices) + len_samp_unlab_embeds, self.n_classes),
            dtype=torch.float32,
            device=self.device
        )

        for idx, label in enumerate(self.lab_embedds_dict['labels']): self.X[idx][label] = 1.
        
        for idx in range(len(self.labeled_indices), len(self.labeled_indices) + len_samp_unlab_embeds):
            for label in range(self.n_classes): self.X[idx][label] = 1. / self.n_classes
        
        
    # correct
    def gtg(self, indices):
        err = float('Inf')
        i = 0
        old_rowsum_X = 0
        
        while err > self.params['gtg_tol'] and i < self.params['gtg_max_iter']:
            X_old = copy.deepcopy(self.X)
            
            self.X *= torch.mm(self.A, self.X)
            
            #------------------------------------------------------------------------------------
            rowsum_X = torch.sum(self.X).item()
            if rowsum_X < old_rowsum_X: # it has to be increasing or at least equal
                raise Exception('Sum of the denominator vectro lower than the previous step')
            old_rowsum_X = rowsum_X
            #------------------------------------------------------------------------------------

            self.X /= torch.sum(self.X, dim=1, keepdim=True)            
        
            iter_entropy = entropy(self.X).to(self.device) # both labeled and sample unlabeled
            # I have to map only the sample_unlabeled to the correct position

            # I iterate only the sampled unlabeled one                
            for idx, unlab_ent_val in zip(indices, iter_entropy[len(self.labeled_indices):]):
                self.entropy_history[idx][i] = unlab_ent_val
            
            err = torch.norm(self.X - X_old)
            i += 1



    def query(self, sample_unlab_subset, n_top_k_obs):
            
        # set the entire batch size to the dimension of the sampled unlabeled set
        unlab_batch_size = len(sample_unlab_subset)

        # we have the batch size which is equal to the number of sampled observation from the unlabeled set                    
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=unlab_batch_size,
            shuffle=False, pin_memory=True
            # set shuffle to false since I do not have interest on shufflind the dataloader, since I have only to get the embeddings
            # thus there is no needs on shuffling the unlabeled dataloader
        )
                
        print(' => Getting the labeled and unlabeled embeddings')
        self.lab_embedds_dict = {'embedds': None, 'labels': None}
        self.unlab_embedds_dict = {'embedds': None}
            
        self.get_embeddings(self.lab_train_dl, self.lab_embedds_dict)
        self.get_embeddings(self.unlab_train_dl, self.unlab_embedds_dict)
        print(' DONE\n')
            
            

        # I save the entropy history in order to be able to plot it
        self.entropy_history = torch.zeros(
            (len(self.transformed_trainset), self.params['gtg_max_iter']), device=self.device
        )
            

        # for each split
        # split_idx -> indices of the split
        # indices -> are the list of indices for the given batch which ARE CONSISTENT SINCE ARE REFERRED TO THE INDEX OF THE ORIGINAL DATASET
        for split_idx, (indices, _, _) in enumerate(self.unlab_train_dl):
            print(f' => Running GTG for split {split_idx}')
            self.get_A(self.unlab_embedds_dict['embedds'][split_idx * unlab_batch_size : (split_idx + 1) * unlab_batch_size])
            self.get_X(indices.shape[0])
            self.gtg(indices)
            self.clear_memory()                
            print(' DONE\n')
            
        
        
        ###########-------------------------- AREA -------------------------###########
        #area_trapz = trapz(-np.diff(self.entropy_history.cpu().numpy(), axis=1), np.arange(self.params['gtg_max_iter'] - 1), axis=0)
        #overall_topk = torch.topk(torch.from_numpy(area_trapz), n_top_k_obs)
        ###############################################################################
        
        
        #----------------------------------------------------------------------------------------------------------------------------------------
        # every check that I've made before say that the implementation is correct thus I think the problem is how I manage the derivatives
        # however logically it seems correct...
        
        
        # absolute value of the derivate to remove the oscilaltions -> observaion that have oscillations means that are difficult too
        self.entropy_pairwise_der = torch.abs(-torch.diff(self.entropy_history, dim=1))


        # plot entropy history
        plot_history(self.entropy_history, f'./app/gtg_entropy/history/{self.method_name}_{self.iter}.png', self.iter, self.params['gtg_max_iter'])

            
        # getting the last column that have at least one element with entropy greater than 1e-15
        for col_index in range(self.entropy_pairwise_der.size(1) - 1, -1, -1):
            if torch.any(self.entropy_pairwise_der[:, col_index] > 1e-15): break
            
            
        # set the weights to increasing value until col_index
        weights = torch.zeros(self.params['gtg_max_iter'] - 1, dtype=torch.float32, device=self.device) 
        weights[:col_index] = torch.flip(torch.linspace(1, 0.1, col_index), [0]).to(self.device)
        
        
        # plot in the entropy derivatives and weighted entropy derivatives
        plot_derivatives(
            self.entropy_pairwise_der,
            self.entropy_pairwise_der * weights,
            f'./app/gtg_entropy/weighted_derivatives/{self.method_name}_{self.iter}.png',
            self.iter,
            self.params['gtg_max_iter'] - 1
        )
        
        # weighted average
        #overall_topk = torch.topk(torch.mean(self.entropy_pairwise_der * weights, dim = 1), n_top_k_obs)
        overall_topk = torch.topk(
            torch.sum(self.entropy_pairwise_der * weights, dim = 1) / col_index,
            n_top_k_obs
        )
        #overall_topk = torch.topk(torch.mean(self.entropy_pairwise_der, dim=1), n_top_k_obs)
        
        # plot in the top k entropy derivatives and weighted entropy derivatives
        plot_derivatives(
            self.entropy_pairwise_der[overall_topk.indices.tolist()],
            (self.entropy_pairwise_der * weights)[overall_topk.indices.tolist()],
            f'./app/gtg_entropy/topk_weighted_derivatives/{self.method_name}_{self.iter}.png',
            self.iter,
            self.params['gtg_max_iter'] - 1
        )
        
        
        self.clear_cuda_variables(
            [self.entropy_pairwise_der, self.entropy_history, self.lab_embedds_dict,
            self.unlab_embedds_dict, weights]
        )
        
        return overall_topk.indices.tolist()
    