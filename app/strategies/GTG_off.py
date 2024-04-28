
from ActiveLearner import ActiveLearner
from utils import plot_tsne_A, create_directory, entropy, plot_gtg_entropy_tensor, Entropy_Strategy

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

import gc
from typing import Dict, Any, List, Tuple
    
import logging
logger = logging.getLogger(__name__)


class GTG_off(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any], gtg_p: Dict[str, Any]) -> None:
        
        self.get_A_fn = {
            'cos_sim': self.get_A_cos_sim,
            'corr': self.get_A_corr,
            'e_d': self.get_A_e_d,
        }

        self.A_function: str = gtg_p['A_function']
        self.ent_strategy: Entropy_Strategy = gtg_p['ent_strategy']
        self.rbf_aff: bool = gtg_p['rbf_aff']
        self.gtg_tol: float = gtg_p['gtg_tol']
        self.gtg_max_iter: int = gtg_p['gtg_max_iter']
        self.strategy_type: str = gtg_p['strategy_type']
        self.threshold_strategy: str = gtg_p['threshold_strategy']
        self.threshold: float = gtg_p['threshold']

        ll_v = 'v2' if 'll_version' in ct_p and ct_p['ll_version'] == 2 else ''
        str_rbf = 'rbf_' if self.rbf_aff else ''
        if self.threshold_strategy != None:
            str_treshold = f'{self.threshold_strategy}_{self.threshold}' if self.threshold_strategy != 'mean' else 'mean'            
            strategy_name = f'{self.__class__.__name__}_{ll_v}_{self.strategy_type}_{str_rbf}{self.A_function}_{self.ent_strategy.name}_{str_treshold}'
        else:
            strategy_name = f'{self.__class__.__name__}_{ll_v}_{self.strategy_type}_{str_rbf}{self.A_function}_{self.ent_strategy.name}'
                
        super().__init__(ct_p, t_p, al_p, strategy_name)
        
        create_directory(self.path + '/gtg_entropies_plots')
        
        
    
    def get_A_treshold(self, A: torch.Tensor) -> Any:
        if self.threshold_strategy == 'mean': return torch.mean(A)
        else: return self.threshold
    
        
    # select the relative choosen affinity matrix method
    def get_A(self) -> None:

        concat_embedds = torch.cat(
            (self.lab_embedds_dict['embedds'], self.unlab_embedds_dict['embedds'])
        ).to(self.device)
        
        del self.unlab_embedds_dict
        torch.cuda.empty_cache()
        
        # compute the affinity matrix
        A = self.get_A_rbfk(concat_embedds, to_cpu=True) if self.rbf_aff else self.get_A_fn[self.A_function](concat_embedds)
    
        initial_A = torch.clone(A)
        
        if self.threshold_strategy != None:
            # remove weak connections with the choosen threshold strategy and value
            logger.info(f' Affinity Matrix Threshold to be used: {self.threshold_strategy}, {self.threshold} -> {self.get_A_treshold(A)}')
            if self.A_function != 'e_d': A = torch.where(A < self.get_A_treshold(A), 0, A)
            else: A = torch.where(A > self.get_A_treshold(A), 1, A)
        
        
        if self.strategy_type == 'diversity':
            # set the whole matrix as a distance matrix and not similarity matrix
            A = 1 - A
        elif self.strategy_type == 'mixed':    
            # set the unlabeled submatrix as distance matrix and not similarity matrix
            n_lab_obs = len(self.labeled_indices)
            
            if self.A_function == 'e_d':
                A[:n_lab_obs, :n_lab_obs] = 1 - A[:n_lab_obs, :n_lab_obs] #LL to similarity
                
            else:
                A[n_lab_obs:, n_lab_obs:] = 1 - A[n_lab_obs:, n_lab_obs:] #UU to distance
                
                A[:n_lab_obs, :n_lab_obs] = 1 - A[:n_lab_obs, :n_lab_obs] #UL to distance
                A[n_lab_obs:, n_lab_obs:] = 1 - A[n_lab_obs:, n_lab_obs:] #LU to distance

        
        self.A = A
        
        # plot the TSNE fo the original and modified affinity matrix
        plot_tsne_A(
            (initial_A, A),
            (self.lab_embedds_dict['labels'], self.unlabeled_labels), self.dataset.classes,
            self.ct_p['timestamp'], self.ct_p['dataset_name'], self.ct_p['trial'], self.strategy_name, self.A_function, self.strategy_type, self.iter
        )
        
        del A
        del initial_A
        del concat_embedds
        torch.cuda.empty_cache()



    def get_A_rbfk(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        
        device = 'cpu' if to_cpu else self.device
        A_matrix = self.get_A_fn[self.A_function](concat_embedds, to_cpu).to(device)

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
        ], device=device), dim=0)
        
        A = torch.exp(-A_matrix.pow(2) / (torch.mm(sigmas.T, sigmas))).to(device)
        A = torch.clamp(A, min=0., max=1.)
        
        del A_matrix
        del sigmas
        torch.cuda.empty_cache()
        
        return A.to(self.device)
    
    
    def get_A_e_d(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        A = torch.cdist(concat_embedds, concat_embedds).to('cpu' if to_cpu else self.device)
        return torch.clamp(A, min=0., max=1.)


    def get_A_cos_sim(self, concat_embedds: torch.Tensor, to_cpu = True) -> torch.Tensor:
        device = 'cpu' if to_cpu else self.device
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(device)
        
        A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(device)
        A.fill_diagonal_(1.)
        A = torch.clamp(A, min=0., max=1.)

        del normalized_embedding
        torch.cuda.empty_cache()
        
        return A.to(self.device)
        
        
    def get_A_corr(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        A = torch.corrcoef(concat_embedds).to('cpu' if to_cpu else self.device)
        return torch.clamp(A, min=0., max=1.)
        

    def get_X(self) -> None:
                
        self.X: torch.Tensor = torch.zeros(
            (len(self.labeled_indices) + self.len_unlab_sample, self.dataset.n_classes),
            dtype=torch.float32, device=self.device
        )

        for idx, label in enumerate(self.lab_embedds_dict['labels']): self.X[idx][int(label.item())] = 1.
        
        for idx in range(len(self.labeled_indices), len(self.labeled_indices) + self.len_unlab_sample):
            for label in range(self.dataset.n_classes): self.X[idx][label] = 1. / self.dataset.n_classes
        
        del self.lab_embedds_dict
        torch.cuda.empty_cache()
        
        
        
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
        
        del X_old
        torch.cuda.empty_cache()


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
            
        # set the entire batch size to the dimension of the sampled unlabeled set
        self.len_unlab_sample = len(sample_unlab_subset)

        # we have the batch size which is equal to the number of sampled observation from the unlabeled set                    
        unlab_train_dl = DataLoader(                           # set shuffle to false since I do not have interest on shufflind the dataloader, since I have only to get the embeddings
            sample_unlab_subset, batch_size=self.len_unlab_sample,  # thus there is no needs on shuffling the unlabeled dataloader
            shuffle=False, pin_memory=True
        )
        lab_train_dl = DataLoader(
            self.labeled_subset, batch_size=self.ds_t_p['batch_size'],  # thus there is no needs on shuffling the unlabeled dataloader
            shuffle=False, pin_memory=True
        )
                
        logger.info(' => Getting the labeled and unlabeled embeddings')
        self.lab_embedds_dict = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=self.device),
            'labels': torch.empty(0, dtype=torch.int8)
        }
        self.unlab_embedds_dict = {'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=self.device)}
        
        self.load_best_checkpoint()
        
        self.get_embeddings(lab_train_dl, self.lab_embedds_dict)
        self.get_embeddings(unlab_train_dl, self.unlab_embedds_dict)
        logger.info(' DONE\n')
            
        # I can take the indices and labels from the dataloader, without using the get_embeddings function so no cuda memory is used in this case
        # I have a single batch
        self.unlabeled_idxs = next(iter(unlab_train_dl))[0]
        self.unlabeled_labels = next(iter(unlab_train_dl))[2]
            
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
        torch.cuda.empty_cache()         
        
                            
        logger.info(f' => Extracting the Top-k unlabeled observations using {self.ent_strategy}')
        
        if self.ent_strategy is Entropy_Strategy.MEAN:
            # computing the mean of the entropis history
            mean_ent = torch.mean(self.unlab_entropy_hist, dim=1)
            overall_topk = torch.topk(mean_ent, k=n_top_k_obs)
          
        elif self.ent_strategy is Entropy_Strategy.H_INT:
            # computing the area of the entropis history using trapezoid formula 
            area = torch.trapezoid(self.unlab_entropy_hist, dim=1)
            overall_topk = torch.topk(area, k=n_top_k_obs, largest=True)
        
            '''elif self.ent_strategy is Entropy_Strategy.DER:
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
                tensor=self.unlab_entropy_der, topk=overall_topk.indices.tolist(), lab_unlabels=self.unlabeled_labels.tolist(), classes=self.dataset.classes, 
                path=self.path, iter=self.iter - 1, max_x=self.gtg_max_iter - 1, dir='derivatives'
            )
        
            del self.unlab_entropy_der
            torch.cuda.empty_cache()'''
        
        else: 
            logger.exception('Unrecognized derivates computation strategy')
            raise Exception('Unrecognized derivates computation strategy')
        
        # plot entropy hisstory tensor
        plot_gtg_entropy_tensor(
            tensor=self.unlab_entropy_hist, topk=overall_topk.indices.tolist(), lab_unlabels=self.unlabeled_labels.tolist(), classes=self.dataset.classes, 
            path=self.path, iter=self.iter - 1, max_x=self.gtg_max_iter, dir='history'
        )
        
            
        del self.unlab_entropy_hist
        gc.collect()
        torch.cuda.empty_cache() 
        
        logger.info(' DONE\n')
        
        return overall_topk.indices.tolist(), [self.unlabeled_idxs[id].item() for id in overall_topk.indices.tolist()]
    