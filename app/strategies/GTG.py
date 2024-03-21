
from strategies.Strategies import Strategies
from utils import create_directory, entropy, plot_history, plot_derivatives, Entropy_Strategy

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from scipy.integrate import simpson #,trapz
import numpy as np

import gc
from typing import Dict, Any, List
    


class GTG(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], training_params: Dict[str, Any], gtg_params: Dict[str, int], LL: bool) -> None:
        
        super().__init__(al_params, training_params, LL)
        
        self.get_A_fn = {
            'cos_sim': self.get_A_cos_sim,
            'corr': self.get_A_corr,
            'e_d': self.get_A_e_d,
        }

        self.A_function: str = gtg_params['A_function']
        self.zero_diag: bool = gtg_params['zero_diag']
        self.ent_strategy: Entropy_Strategy = gtg_params['ent_strategy']
        self.rbf_aff: bool = gtg_params['rbf_aff']
        self.gtg_tol: float = gtg_params['gtg_tol']
        self.gtg_max_iter: int = gtg_params['gtg_max_iter']
        self.strategy_type: str = gtg_params['strategy_type']
                
        str_diag = '0diag' if self.zero_diag else '1diag'
        rbf_str = 'rbf_' if self.rbf_aff else ''
                
        self.method_name = f'{self.__class__.__name__}_{self.strategy_type}_{rbf_str}{self.A_function}_{self.ent_strategy.name}_{str_diag}_LL' if LL \
            else f'{self.__class__.__name__}_{self.strategy_type}_{rbf_str}{self.A_function}_{self.ent_strategy.name}_{str_diag}'
                            
    
        
    # select the relative choosen affinity matrix method
    def get_A(self, samp_unlab_embeddings: torch.Tensor) -> None:
        concat_embedds = torch.cat((self.lab_embedds_dict['embedds'], samp_unlab_embeddings)).to(self.device)
        
        # compute the affinity matrix
        A = self.get_A_fn[self.A_function](concat_embedds) \
        if self.A_function != 'rbfk' else self.get_A_rbfk(concat_embedds)
        
        if self.strategy_type == 'diversity':
            # set the whole matrix as a distance matrix and not similarity matrix
            A = 1 - A
        elif self.strategy_type == 'mixed':
            # set the unlabeled submatrix as distance matrix and not similarity matrix
            n_lab_obs = len(self.lab_embedds_dict['embedds'])
            A[:n_lab_obs, n_lab_obs:] = 1 - A[:n_lab_obs, n_lab_obs:]
            A[n_lab_obs:, :n_lab_obs] = 1 - A[n_lab_obs:, :n_lab_obs]
            
            # Unlabeled VS Unlabeled -> similarity = A
            # Labeled VS Labeled -> similarity = A
            # Unlabeled VS Labeled -> distance = 1 - A
            # Labeled VS Unlabeled -> distance = 1 - A
            
        self.A = A


    # should be correct
    def get_A_rbfk(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        
        #e_d = torch.cdist(concat_embedds, concat_embedds).to(self.device)
        aff_matrix = self.get_A_fn[self.A_function](concat_embedds)
        
        if self.A_function == 'e_d':
            # if euclidean distance is choosen we take the 7th smallest observation which is the 7th closest one (ascending order)
            seventh_neigh = concat_embedds[torch.argsort(aff_matrix, dim=1)[:, 6]]            
        else:
            # if correlation or cosine_similairty are choosen we take the 7th highest observation which is the 7th most similar one (descending order)
            seventh_neigh = concat_embedds[torch.argsort(aff_matrix, dim=1, descending=True)[:, 6]]
        
        sigmas = torch.unsqueeze(torch.tensor([
            self.get_A_fn[self.A_function](
                torch.unsqueeze(concat_embedds[i], dim=0), 
                torch.unsqueeze(seventh_neigh[i], dim=0)
            )
            for i in range(concat_embedds.shape[0])
        ], device=self.device), dim=0)
        
        A = torch.exp(-aff_matrix.pow(2) / (torch.mm(sigmas.T, sigmas))).to(self.device)
        
        if self.zero_diag: A.fill_diagonal_(0.)  
              
        return A
    
    
    
    # correct
    def get_A_e_d(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        
        A = torch.cdist(concat_embedds, concat_embedds).to(self.device)
        
        if self.zero_diag: A.fill_diagonal_(0.)
        else: A.fill_diagonal_(1.)
        return A



    # correct
    def get_A_cos_sim(self, concat_embedds: torch.Tensor) -> torch.Tensor:
                                
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(self.device)
        
        A = F.relu(torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device))
        
        if self.zero_diag: A.fill_diagonal_(0.)
        else: A.fill_diagonal_(1.)
        
        return A
        
        
        
    # correct
    def get_A_corr(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        
        A = F.relu(torch.corrcoef(concat_embedds).to(self.device))
        
        if self.zero_diag: A.fill_diagonal_(0.)
        
        return A

        

    # correct
    def get_X(self, len_samp_unlab_embeds: torch.Tensor) -> None:
        
        self.X: torch.Tensor = torch.zeros(
            (len(self.labeled_indices) + len_samp_unlab_embeds, self.n_classes),
            dtype=torch.float32,
            device=self.device
        )

        for idx, label in enumerate(self.lab_embedds_dict['labels']): self.X[idx][label] = 1.
        
        for idx in range(len(self.labeled_indices), len(self.labeled_indices) + len_samp_unlab_embeds):
            for label in range(self.n_classes): self.X[idx][label] = 1. / self.n_classes
        
        
        
    # correct
    def gtg(self, indices: List[int]) -> None:       
        
        err = float('Inf')
        i = 0
        old_rowsum_X = 0
        
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = torch.clone(self.X)
            
            self.X *= torch.mm(self.A, self.X)
            
            rowsum_X = torch.sum(self.X).item()
            if rowsum_X < old_rowsum_X: # it has to be increasing or at least equal
                raise Exception('Sum of the vector on the denominator is lower than the previous step')
            old_rowsum_X = rowsum_X
            
            self.X /= torch.sum(self.X, dim=1, keepdim=True)            
        
            iter_entropy = entropy(self.X).to(self.device) # both labeled and sample unlabeled
            # I have to map only the sample_unlabeled to the correct position

            # I iterate only the sampled unlabeled one                
            for idx, unlab_ent_val in zip(indices, iter_entropy[len(self.labeled_indices):]):
                self.entropy_history[idx][i] = unlab_ent_val
            
            err = torch.norm(self.X - X_old)
            i += 1
            


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> List[int]:
        #gc.collect()
        #torch.cuda.empty_cache()
            
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
            (len(self.transformed_trainset), self.gtg_max_iter), device=self.device
        )
        

        # for each split
        # split_idx -> indices of the split
        # indices -> are the list of indices for the given batch which ARE CONSISTENT SINCE ARE REFERRED TO THE INDEX OF THE ORIGINAL DATASET
        for split_idx, (indices, _, _) in enumerate(self.unlab_train_dl):
            print(f' => Running GTG for split {split_idx}')
            self.get_A(self.unlab_embedds_dict['embedds'][split_idx * unlab_batch_size : (split_idx + 1) * unlab_batch_size])
            self.get_X(indices.shape[0])
            self.gtg(indices)
            
            del self.A
            del self.X
            torch.cuda.empty_cache()
            
            print(' DONE\n')
            
        del self.lab_embedds_dict
        del self.unlab_embedds_dict
        torch.cuda.empty_cache()         
        
        path = f'results/{self.timestamp}/{self.dataset_name}/{self.samp_iter}/gtg_entropies_plots'
        create_directory(path)
        
            
        if self.ent_strategy is Entropy_Strategy.LAST:
            # returning the last entropies values
            overall_topk = torch.topk(self.entropy_history[-1], n_top_k_obs)
        
        if self.ent_strategy is Entropy_Strategy.HISTORY_INTEGRAL:
            # computing the area of each entropies derivates fucntion via the trapezius formula 
            #area: np.ndarray = trapz(-np.diff(self.entropy_history.cpu().numpy(), axis=1))
            #area: np.ndarray = simpson(-np.diff(self.entropy_history.cpu().numpy(), axis=1))
            
            #-----------------------------------------------------------------------------------------------
            # WRONG COMPUTING THE AREA OVER THE DERIVATIVES OF THE ENTROPY
            # I WANT THE ARE OF THE ORIGINAL FUNCTION ENTROPIES THUS THE THEIR HISOTRY OVER GTG ITERATIONS
            area: np.ndarray = simpson(self.entropy_history.cpu().numpy())
            #-----------------------------------------------------------------------------------------------
                        
            overall_topk = torch.topk(torch.from_numpy(area), n_top_k_obs)
        
        else:
            # absolute value of the derivate to remove the oscilaltions -> observaion that have oscillations means that are difficult too
            self.entropy_pairwise_der = torch.abs(-torch.diff(self.entropy_history, dim=1))


            create_directory(f'{path}/{self.method_name}')
            create_directory(f'{path}/{self.method_name}/history')

            
            # plot entropy history
            plot_history(self.entropy_history, f'{path}/{self.method_name}/history/{self.iter - 1}.png', self.iter - 1, self.gtg_max_iter)

            if self.ent_strategy is Entropy_Strategy.WEIGHTED_AVERAGE_DERIVATIVES:
                # getting the last column that have at least one element with entropy greater than 1e-15
                for col_index in range(self.entropy_pairwise_der.size(1) - 1, -1, -1):
                    if torch.any(self.entropy_pairwise_der[:, col_index] > 1e-15): break
                    
                    
                # set the weights to increasing value until col_index
                weights = torch.zeros(self.gtg_max_iter - 1, dtype=torch.float32, device=self.device) 
                weights[:col_index] = torch.flip(torch.linspace(1, 0.1, col_index), [0]).to(self.device)
                
                
                # weighted average excluding the derivates that are bellow the threshold
                overall_topk = torch.topk(
                    torch.sum(self.entropy_pairwise_der * weights, dim = 1) / col_index,
                    n_top_k_obs
                )    
            
            elif self.ent_strategy is Entropy_Strategy.MEAN_DERIVATIVES:
                # classic average among the derivates
                overall_topk = torch.topk(torch.mean(self.entropy_pairwise_der, dim=1), n_top_k_obs)
            else:
                raise Exception('Unrecognized derivates computation strategy')


            create_directory(f'{path}/{self.method_name}')
            create_directory(f'{path}/{self.method_name}/weighted_derivatives')
            
            # plot in the entropy derivatives and weighted entropy derivatives
            plot_derivatives(
                self.entropy_pairwise_der,
                self.entropy_pairwise_der * weights if self.ent_strategy is Entropy_Strategy.WEIGHTED_AVERAGE_DERIVATIVES else self.entropy_pairwise_der,
                f'{path}/{self.method_name}/weighted_derivatives/{self.iter - 1}.png',
                self.iter - 1, self.gtg_max_iter - 1
            )
            
            
            create_directory(f'{path}/{self.method_name}/topk_weighted_derivatives')
            
            # plot in the top k entropy derivatives and weighted entropy derivatives
            plot_derivatives(
                self.entropy_pairwise_der[overall_topk.indices.tolist()],
                (self.entropy_pairwise_der * weights if self.ent_strategy is Entropy_Strategy.WEIGHTED_AVERAGE_DERIVATIVES else self.entropy_pairwise_der)[overall_topk.indices.tolist()],
                f'{path}/{self.method_name}/topk_weighted_derivatives/{self.iter - 1}.png',
                self.iter - 1, self.gtg_max_iter - 1
            )
            
        
            del self.entropy_pairwise_der
            torch.cuda.empty_cache()  
            
        del self.entropy_history
        gc.collect()
        torch.cuda.empty_cache() 
        
        return overall_topk.indices.tolist()