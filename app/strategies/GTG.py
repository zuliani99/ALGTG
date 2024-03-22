from strategies.Strategies import Strategies
from utils import plot_tsne_A, create_directory, entropy, plot_history, plot_derivatives, Entropy_Strategy

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from scipy.integrate import simpson #,trapz
from sklearn.metrics.pairwise import cosine_similarity
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
    def get_A(self) -> None:
        
        concat_embedds = torch.cat(
            (self.lab_embedds_dict['embedds'], self.unlab_embedds_dict['embedds'])
        ).to(self.device)
        
        # compute the affinity matrix
        A = self.get_A_fn[self.A_function](concat_embedds) \
        if self.A_function != 'rbfk' else self.get_A_rbfk(concat_embedds)

        A_1 = torch.clone(A)
        
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
        
        # remove weak connections 
        A_2 = torch.where(A < 0.5, 0, A)
            
        self.A = A_2
        
        # plot the TSNE fo the original and modified affinity matrix
        plot_tsne_A(
            (A_1, A_2),
            (self.lab_embedds_dict['labels'], self.labels_unlabeled), self.classes,
            self.timestamp, self.dataset_name, self.samp_iter, self.method_name, self.A_function, self.strategy_type
        )



    # should be correct
    def get_A_rbfk(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        
        A_matrix = self.get_A_fn[self.A_function](concat_embedds)
        
        if self.A_function == 'e_d':
            # if euclidean distance is choosen we take the 7th smallest observation which is the 7th closest one (ascending order)
            seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1)[:, 6]]            
        else:
            # if correlation or cosine_similairty are choosen we take the 7th highest observation which is the 7th most similar one (descending order)
            seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1, descending=True)[:, 6]]
        
        sigmas = torch.unsqueeze(torch.tensor([
            self.get_A_fn[self.A_function](
                torch.unsqueeze(concat_embedds[i], dim=0), 
                torch.unsqueeze(seventh_neigh[i], dim=0)
            )
            for i in range(concat_embedds.shape[0])
        ], device=self.device), dim=0)
        
        A = torch.exp(-A_matrix.pow(2) / (torch.mm(sigmas.T, sigmas))).to(self.device)
        
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
                                
        #normalized_embedding = F.normalize(concat_embedds, dim=-1).to(self.device)
        
        #A = F.relu(torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device))
        
        #if self.zero_diag: A.fill_diagonal_(0.)
        #else: A.fill_diagonal_(1.)
        A = cosine_similarity(concat_embedds.cpu().numpy())
        return torch.from_numpy(A).to(self.device)
        #return A
        
        
        
    # correct
    def get_A_corr(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        
        A = F.relu(torch.corrcoef(concat_embedds).to(self.device))
        
        if self.zero_diag: A.fill_diagonal_(0.)
        return A

        

    # correct
    def get_X(self) -> None:
        
        len_samp_unlab_embeds = len(self.idx_unlabeled)
        
        self.X: torch.Tensor = torch.zeros(
            (len(self.labeled_indices) + len_samp_unlab_embeds, self.n_classes),
            dtype=torch.float32,
            device=self.device
        )

        for idx, label in enumerate(self.lab_embedds_dict['labels']): self.X[idx][label.item()] = 1.
        
        for idx in range(len(self.labeled_indices), len(self.labeled_indices) + len_samp_unlab_embeds):
            for label in range(self.n_classes): self.X[idx][label] = 1. / self.n_classes
        
        
        
    def check_increasing_sum(self, old_rowsum_X):
        rowsum_X = torch.sum(self.X).item()
        if rowsum_X < old_rowsum_X: # it has to be increasing or at least equal
            raise Exception('Sum of the vector on the denominator is lower than the previous step')
        return rowsum_X
        
        
        
    # correct
    def graph_trasduction_game(self) -> None:
        
        self.get_A()
        self.get_X()
        
        err = float('Inf')
        i = 0
        old_rowsum_X = 0
        
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = torch.clone(self.X)
            
            self.X *= torch.mm(self.A, self.X)
            
            old_rowsum_X = self.check_increasing_sum(old_rowsum_X)
            
            self.X /= torch.sum(self.X, dim=1, keepdim=True)            
        
            iter_entropy = entropy(self.X).to(self.device) # there are both labeled and sample unlabeled
            # I have to map only the sample_unlabeled to the correct position
            
            # Update only the unlabeled observations
            assert len(self.entropy_history_unlabeled[:,i]) == len(iter_entropy[len(self.labeled_indices):]), 'should have the same dimension'
            self.entropy_history_unlabeled[:,i] = iter_entropy[len(self.labeled_indices):]
            # they have the same dimension
                        
            err = torch.norm(self.X - X_old)
            i += 1
            


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> List[int]:
        #gc.collect()
        #torch.cuda.empty_cache()
            
        # set the entire batch size to the dimension of the sampled unlabeled set
        unlab_batch_size = len(sample_unlab_subset)

        # we have the batch size which is equal to the number of sampled observation from the unlabeled set                    
        self.unlab_train_dl = DataLoader(                                       # set shuffle to false since I do not have interest on shufflind the dataloader, since I have only to get the embeddings
            sample_unlab_subset, batch_size=unlab_batch_size,                   # thus there is no needs on shuffling the unlabeled dataloader
            shuffle=False, pin_memory=True
            
        )
                
        print(' => Getting the labeled and unlabeled embeddings')
        self.lab_embedds_dict = {'embedds': None, 'labels': None}
        self.unlab_embedds_dict = {'embedds': None}#, 'labels': None, 'idxs': None}
            
        self.get_embeddings(self.lab_train_dl, self.lab_embedds_dict)
        self.get_embeddings(self.unlab_train_dl, self.unlab_embedds_dict)
        print(' DONE\n')
            
        # without using cuda memory
        self.idx_unlabeled = next(iter(self.unlab_train_dl))[0]
        self.labels_unlabeled = next(iter(self.unlab_train_dl))[2]
            
            
        # I save the entropy history in order to be able to plot it
        self.entropy_history_unlabeled = torch.zeros((unlab_batch_size, self.gtg_max_iter), device=self.device)
        
        #labels: List[int] = self.lab_embedds_dict['labels'].cpu().tolist()
        #labels.extend(self.labels_unlabeled.tolist())
            
        print(' => Execution of the Graph Trasduction Game')
        self.graph_trasduction_game()
        print(' DONE\n')
        
        
        del self.A
        del self.X
        del self.lab_embedds_dict
        del self.unlab_embedds_dict
        torch.cuda.empty_cache()         
        
        path = f'results/{self.timestamp}/{self.dataset_name}/{self.samp_iter}/gtg_entropies_plots'
        create_directory(path)
        
        print(f' => Extracting the Top-k unlabeled observation using {self.ent_strategy}')
            
        if self.ent_strategy is Entropy_Strategy.LAST:
            # returning the last entropies values
            overall_topk = torch.topk(self.entropy_history_unlabeled[-1], n_top_k_obs)
        
        if self.ent_strategy is Entropy_Strategy.H_INT:
            # computing the area of each entropies derivates fucntion via the trapezius formula 
            #area: np.ndarray = trapz(-np.diff(self.entropy_history.cpu().numpy(), axis=1))
            #area: np.ndarray = simpson(-np.diff(self.entropy_history.cpu().numpy(), axis=1))
            
            #-----------------------------------------------------------------------------------------------
            # WRONG COMPUTING THE AREA OVER THE DERIVATIVES OF THE ENTROPY
            # I WANT THE ARE OF THE ORIGINAL FUNCTION ENTROPIES THUS THE THEIR HISOTRY OVER GTG ITERATIONS
            area: np.ndarray = simpson(self.entropy_history_unlabeled.cpu().numpy())
            #-----------------------------------------------------------------------------------------------
                        
            overall_topk = torch.topk(torch.from_numpy(area), n_top_k_obs)
        
        else:
            # absolute value of the derivate to remove the oscilaltions -> observaion that have oscillations means that are difficult too
            self.entropy_pairwise_unlabeled_der = torch.abs(-torch.diff(self.entropy_history_unlabeled, dim=1))
            
            # plot entropy history
            plot_history(path, self.labels_unlabeled.tolist(), self.classes, self.method_name, self.entropy_history_unlabeled, self.iter - 1, self.gtg_max_iter)

            if self.ent_strategy is Entropy_Strategy.W_A_DER:
                # getting the last column that have at least one element with entropy greater than 1e-15
                for col_index in range(self.entropy_pairwise_unlabeled_der.size(1) - 1, -1, -1):
                    if torch.any(self.entropy_pairwise_unlabeled_der[:, col_index] > 1e-15): break
                    
                    
                # set the weights to increasing value until col_index
                weights = torch.zeros(self.gtg_max_iter - 1, dtype=torch.float32, device=self.device) 
                weights[:col_index] = torch.flip(torch.linspace(1, 0.1, col_index), [0]).to(self.device)
                
                
                # weighted average excluding the derivates that are bellow the threshold
                overall_topk = torch.topk(
                    torch.sum(self.entropy_pairwise_unlabeled_der * weights, dim = 1) / col_index,
                    n_top_k_obs
                )    
            
            elif self.ent_strategy is Entropy_Strategy.MEAN_DER:
                # classic average among the derivates
                overall_topk = torch.topk(torch.mean(self.entropy_pairwise_unlabeled_der, dim=1), n_top_k_obs)
            else:
                raise Exception('Unrecognized derivates computation strategy')
            
            # plot in the entropy derivatives and weighted entropy derivatives
            plot_derivatives(
                self.method_name, self.entropy_pairwise_unlabeled_der,
                self.entropy_pairwise_unlabeled_der * weights if self.ent_strategy is Entropy_Strategy.W_A_DER else self.entropy_pairwise_unlabeled_der,
                path, self.labels_unlabeled.tolist(), self.classes, self.iter - 1, self.gtg_max_iter - 1, 'weighted_derivatives'
            )
            
            # plot in the top k entropy derivatives and weighted entropy derivatives
            plot_derivatives(
                self.method_name, self.entropy_pairwise_unlabeled_der[overall_topk.indices.tolist()],
                (self.entropy_pairwise_unlabeled_der * weights if self.ent_strategy is Entropy_Strategy.W_A_DER else self.entropy_pairwise_unlabeled_der)[overall_topk.indices.tolist()],
                path, self.labels_unlabeled.tolist(), self.classes, self.iter - 1, self.gtg_max_iter - 1, 'topk_weighted_derivatives'
            )
            
        
            del self.entropy_pairwise_unlabeled_der
            torch.cuda.empty_cache()  
            
        del self.entropy_history_unlabeled
        gc.collect()
        torch.cuda.empty_cache() 
        
        print(' DONE\n')
        
        #return overall_topk.indices.tolist()
        return [self.idx_unlabeled[id].item() for id in overall_topk.indices.tolist()]