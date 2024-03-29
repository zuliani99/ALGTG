from strategies.Strategies import Strategies
from utils import plot_tsne_A, create_directory, entropy, plot_gtg_entropy_tensor, Entropy_Strategy

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from scipy.integrate import simpson
import scipy.spatial as sp
#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import gc
from typing import Dict, Any, List, Tuple
    
import logging
logger = logging.getLogger(__name__)


class GTG(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], training_params: Dict[str, Any], gtg_params: Dict[str, int], LL: bool) -> None:
        
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
        self.threshold_strategy: str = gtg_params['threshold_strategy']
        self.threshold: str = gtg_params['threshold']
                
        str_diag = '0diag' if self.zero_diag else '1diag'
        str_rbf = 'rbf_' if self.rbf_aff else ''
        if self.threshold_strategy != None:
            str_treshold = f'{self.threshold_strategy}_{self.threshold}' if self.threshold_strategy != 'mean' else 'mean'
        
            self.method_name = f'{self.__class__.__name__}_{self.strategy_type}_{str_rbf}{self.A_function}_{self.ent_strategy.name}_{str_diag}_{str_treshold}_LL' if LL \
                else f'{self.__class__.__name__}_{self.strategy_type}_{str_rbf}{self.A_function}_{self.ent_strategy.name}_{str_diag}_{str_treshold}'
        
        else:
            self.method_name = f'{self.__class__.__name__}_{self.strategy_type}_{str_rbf}{self.A_function}_{self.ent_strategy.name}_{str_diag}_LL' if LL \
                else f'{self.__class__.__name__}_{self.strategy_type}_{str_rbf}{self.A_function}_{self.ent_strategy.name}_{str_diag}'
                

        super().__init__(al_params, training_params, LL)
        
        create_directory(self.path + '/gtg_entropies_plots')
        
        
    
    def get_A_treshold(self, A: torch.Tensor) -> float:
        if self.threshold_strategy == 'mean': return torch.mean(A)
        elif self.threshold_strategy == 'threshold': return self.threshold
        else: return np.quantile(A.cpu().numpy(), self.threshold)
    
        
    # select the relative choosen affinity matrix method
    def get_A(self) -> None:

        concat_embedds = torch.cat(
            (self.lab_embedds_dict['embedds'], self.unlab_embedds_dict['embedds'])
        ).to(self.device)
        
        # compute the affinity matrix
        A = self.get_A_rbfk(concat_embedds) if self.rbf_aff else self.get_A_fn[self.A_function](concat_embedds)

        A_1 = torch.clone(A)
        
        if self.strategy_type == 'diversity':
            # set the whole matrix as a distance matrix and not similarity matrix
            A = 1 - A
        elif self.strategy_type == 'mixed':    
            # set the unlabeled submatrix as distance matrix and not similarity matrix
            n_lab_obs = len(self.labeled_indices)
                
            if self.A_function == 'e_d':
                # not working, maybe
                A[:n_lab_obs, :n_lab_obs] = 1 - A[:n_lab_obs, :n_lab_obs] #LL -> similarity
                A[n_lab_obs:, n_lab_obs:] = 1 - A[n_lab_obs:, n_lab_obs:] #UU -> similarity
            else:
                A[:n_lab_obs, n_lab_obs:] = 1 - A[:n_lab_obs, n_lab_obs:] #UL -> distacne
                A[n_lab_obs:, :n_lab_obs] = 1 - A[n_lab_obs:, :n_lab_obs] #LU -> distance
            
            '''if self.A_function == 'e_d':
                A[:n_lab_obs, :n_lab_obs] = 1 - A[:n_lab_obs, :n_lab_obs] #LL to similarity
                
            else:
                A[n_lab_obs:, n_lab_obs:] = 1 - A[n_lab_obs:, n_lab_obs:] #UU to distance
                
                A[:n_lab_obs, :n_lab_obs] = 1 - A[:n_lab_obs, :n_lab_obs] #UL to distance
                A[n_lab_obs:, n_lab_obs:] = 1 - A[n_lab_obs:, n_lab_obs:] #LU to distance'''
                
            
            # Unlabeled VS Unlabeled -> similarity = A
            # Labeled VS Labeled -> similarity = A
            # Unlabeled VS Labeled -> distance = 1 - A
            # Labeled VS Unlabeled -> distance = 1 - A
        
        # remove weak connections with the choosen threshold strategy and value
        #logger.info(f' Affinity Matrix Threshold to be used: {self.threshold_strategy}, {self.threshold} -> {self.get_A_treshold(A)}')
        #A_2 = torch.where(A < self.get_A_treshold(A), 0, A)
        #mat_cos_sim = nn.CosineSimilarity(dim=0)
        #logger.info(f' Cosine Similarity between the initial matrix and the thresholded one: {mat_cos_sim(A_1.flatten(), A_2.flatten()).item()}')

        self.A = A#A_2
        
        # plot the TSNE fo the original and modified affinity matrix
        plot_tsne_A(
            #(A_1, A_2),
            (A_1, A),
            (self.lab_embedds_dict['labels'], self.unlabeled_labels), self.classes,
            self.timestamp, self.dataset_name, self.samp_iter, self.method_name, self.A_function, self.strategy_type, self.iter
        )



    def get_A_rbfk(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        
        A_matrix = self.get_A_fn[self.A_function](concat_embedds)
        
        if self.A_function == 'e_d':
            # if euclidean distance is choosen we take the 7th smallest observation which is the 7th closest one (ascending order)
            seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1)[:, 6]]            
        else:
            # if correlation or cosine_similairty are choosen we take the 7th highest observation which is the 7th most similar one (descending order)
            seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1, descending=True)[:, 6]]
            
        sigmas = torch.unsqueeze(torch.tensor([
            self.get_A_fn[self.A_function](torch.cat((
                torch.unsqueeze(concat_embedds[i], dim=0), torch.unsqueeze(seventh_neigh[i], dim=0)
            )))[0,1].item()
            for i in range(concat_embedds.shape[0]) 
        ], device=self.device), dim=0)
        
        A = torch.exp(-A_matrix.pow(2) / (torch.mm(sigmas.T, sigmas))).to(self.device)
        
        if self.zero_diag: A.fill_diagonal_(0.)  
        return A
    
    
    
    def get_A_e_d(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        
        A = torch.cdist(concat_embedds, concat_embedds).to(self.device)
        
        #if self.zero_diag: A.fill_diagonal_(0.)
        #else: A.fill_diagonal_(1.)
        return A



    def get_A_cos_sim(self, concat_embedds: torch.Tensor) -> torch.Tensor:
                               
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(self.device)
        
        A = F.relu(torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device))
        
        if self.zero_diag: A.fill_diagonal_(0.)
        else: A.fill_diagonal_(1.)
        #A = cosine_similarity(concat_embedds.cpu().numpy())
        #return torch.from_numpy(A).to(self.device)
        return A
        
        
        
    def get_A_corr(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        
        A = F.relu(torch.corrcoef(concat_embedds).to(self.device))
        
        if self.zero_diag: A.fill_diagonal_(0.)
        return A

        

    def get_X(self) -> None:
                
        self.X: torch.Tensor = torch.zeros(
            (len(self.labeled_indices) + self.len_unlab_sample, self.n_classes),
            dtype=torch.float32, device=self.device
        )

        for idx, label in enumerate(self.lab_embedds_dict['labels']): self.X[idx][label.item()] = 1.
        
        for idx in range(len(self.labeled_indices), len(self.labeled_indices) + self.len_unlab_sample):
            for label in range(self.n_classes): self.X[idx][label] = 1. / self.n_classes
        
        
        
    def check_increasing_sum(self, old_rowsum_X):
        
        rowsum_X = torch.sum(self.X).item()
        if rowsum_X < old_rowsum_X: # it has to be increasing or at least equal
            logger.exception('Sum of the vector on the denominator is lower than the previous step')
            raise Exception('Sum of the vector on the denominator is lower than the previous step')
        
        return rowsum_X
        
        
        
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
            
            try:
                # they should have the same dimension
                assert len(self.unlab_entropy_hist[:, i]) == len(iter_entropy[len(self.labeled_indices):])
                # Update only the unlabeled observations
                self.unlab_entropy_hist[:, i] = iter_entropy[len(self.labeled_indices):]
            except AssertionError as err:
                logger.exception('Should have the same dimension')
                raise err
                        
            err = torch.norm(self.X - X_old)
            i += 1
            


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
            
        # set the entire batch size to the dimension of the sampled unlabeled set
        self.len_unlab_sample = len(sample_unlab_subset)

        # we have the batch size which is equal to the number of sampled observation from the unlabeled set                    
        self.unlab_train_dl = DataLoader(                           # set shuffle to false since I do not have interest on shufflind the dataloader, since I have only to get the embeddings
            sample_unlab_subset, batch_size=self.len_unlab_sample,  # thus there is no needs on shuffling the unlabeled dataloader
            shuffle=False, pin_memory=True
        )
                
        logger.info(' => Getting the labeled and unlabeled embeddings')
        self.lab_embedds_dict = {'embedds': None, 'labels': None}
        self.unlab_embedds_dict = {'embedds': None}
            
        self.get_embeddings(self.lab_train_dl, self.lab_embedds_dict)
        self.get_embeddings(self.unlab_train_dl, self.unlab_embedds_dict)
        logger.info(' DONE\n')
            
        # I can take the indices and labels from the dataloader, without using the get_embeddings function so no cuda memory is used in this case
        # I have a single batch
        self.unlabeled_idxs = next(iter(self.unlab_train_dl))[0]
        self.unlabeled_labels = next(iter(self.unlab_train_dl))[2]
            
            
        # I save the entropy history in order to be able to plot it
        self.unlab_entropy_hist = torch.zeros((self.len_unlab_sample, self.gtg_max_iter), device=self.device)
        
            
        logger.info(' => Execution of the Graph Trasduction Game')
        self.graph_trasduction_game()
        logger.info(' DONE\n')
        
        unlab_pred_labels_gtg = torch.argmax(self.X[len(self.labeled_indices):], dim=1).cpu()
        
        # TRUE / FALSE np.ndarray
        self.gtg_result_prediction = (unlab_pred_labels_gtg == self.unlabeled_labels).numpy()
        
        logging.info(f' Unlabeled GTG Accuracty Score: {(unlab_pred_labels_gtg == self.unlabeled_labels).sum().item() / len(unlab_pred_labels_gtg)}')
        
        del self.A
        del self.X
        del self.lab_embedds_dict
        del self.unlab_embedds_dict
        torch.cuda.empty_cache()         
        
                            
        logger.info(f' => Extracting the Top-k unlabeled observations using {self.ent_strategy}')
            
        if self.ent_strategy is Entropy_Strategy.H_INT:
            # computing the area of each entropies derivates fucntion via the simpson formula 
            area: np.ndarray = simpson(self.unlab_entropy_hist.cpu().numpy(), axis=1)
                        
            overall_topk = torch.topk(torch.from_numpy(area), k=n_top_k_obs, largest=True)
        
        elif self.ent_strategy is Entropy_Strategy.DER:
            # compute the pairwise differences to obtaion the entropy derivatives
            self.unlab_entropy_der = -torch.diff(self.unlab_entropy_hist, dim=1)
            
            negative_indices = (self.unlab_entropy_der < 0).nonzero()
            rows, cols = negative_indices.unbind(1)
            
            first_negative = torch.full((1, self.len_unlab_sample), -1).squeeze()
            last_negative = torch.full((1, self.len_unlab_sample), 0).squeeze()
            
            for row, col in zip(rows, cols):
                if first_negative[row] == -1: first_negative[row] = col
                last_negative[row] = col
            
            # setting the range between the first and the last negative cell to negative
            for row, (first_0, last_0) in enumerate(zip(first_negative, last_negative)):
                for idx in range(first_0 + 1, last_0):
                    self.unlab_entropy_der[row, idx] = -torch.abs(self.unlab_entropy_der[row, idx])
                    
            
            bool_ent_der = self.unlab_entropy_der <= 1e-3
            bool_ent_his = self.unlab_entropy_hist[:, 1:] <= 1e-3
            
            denominator = torch.logical_and(bool_ent_der, bool_ent_his)
            denominator = torch.argmax(denominator.long(), dim=1)
            denominator = torch.where(denominator == 0, self.unlab_entropy_der.shape[1], denominator)
            
            
            # computing the actual mean
            overall_topk = torch.topk(
                torch.sum(self.unlab_entropy_der, dim=1) / denominator,
                k=n_top_k_obs, largest=False
            )            

            # plot entropy derivatives tensor
            plot_gtg_entropy_tensor(
                tensor=self.unlab_entropy_der, topk=overall_topk.indices.tolist(), lab_unlabels=self.unlabeled_labels.tolist(), classes=self.classes, 
                path=self.path, iter=self.iter - 1, max_x=self.gtg_max_iter - 1, dir='derivatives'
            )
        
            del self.unlab_entropy_der
            torch.cuda.empty_cache()
        
        else: 
            logger.exception('Unrecognized derivates computation strategy')
            raise Exception('Unrecognized derivates computation strategy')
        
        # plot entropy hisstory tensor
        plot_gtg_entropy_tensor(
            tensor=self.unlab_entropy_hist, topk=overall_topk.indices.tolist(), lab_unlabels=self.unlabeled_labels.tolist(), classes=self.classes, 
            path=self.path, iter=self.iter - 1, max_x=self.gtg_max_iter, dir='history'
        )
        
            
        del self.unlab_entropy_hist
        gc.collect()
        torch.cuda.empty_cache() 
        
        logger.info(' DONE\n')
        
        return overall_topk.indices.tolist(), [self.unlabeled_idxs[id].item() for id in overall_topk.indices.tolist()]
    