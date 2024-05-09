
from ActiveLearner import ActiveLearner
from utils import plot_tsne_A, create_directory, entropy, plot_gtg_entropy_tensor

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

import gc
import os
from typing import Dict, Any, List, Tuple
    
import logging
logger = logging.getLogger(__name__)


class GTG_off(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any], gtg_p: Dict[str, Any]) -> None:
        
        self.get_A_fn = {
            'cos_sim': self.get_A_cos_sim,
            'corr': self.get_A_corr,
            'rbfk': self.get_A_rbfk,
        }

        self.gtg_tol: float = gtg_p['gtg_t']
        self.gtg_max_iter: int = gtg_p['gtg_i']
        
        self.AM_function: str = gtg_p['am']
        self.AM_strategy: str = gtg_p['am_s']
        self.AM_threshold_strategy: str = gtg_p['am_ts']
        self.AM_threshold: float = gtg_p['am_t']
        
        self.ent_strategy: str = gtg_p['e_s']

        if self.AM_threshold_strategy != None:
            str_tresh_strat = f'{self.AM_threshold_strategy}_{self.AM_threshold}' if self.AM_threshold_strategy != 'mean' else 'ts-mean'            
            strategy_name = f'{self.__class__.__name__}_{self.AM_function}_{self.AM_strategy}_{str_tresh_strat}_es-{self.ent_strategy}'
        else:
            strategy_name = f'{self.__class__.__name__}_{self.AM_function}_{self.AM_strategy}_es-{self.ent_strategy}'
                
        super().__init__(ct_p, t_p, al_p, strategy_name)
        
        create_directory(self.path + '/gtg_entropies_plots')
        
        
    
    def get_A_treshold(self, A: torch.Tensor) -> Any:
        if self.AM_threshold_strategy == 'mean': return torch.mean(A)
        else: return self.AM_threshold
    
        
    # select the relative choosen affinity matrix method
    def get_A(self) -> None:
        
        logger.info(' => Computing Affinity Matrix...')

        concat_embedds = torch.cat((self.lab_embedds_dict['embedds'], self.unlab_embedds_dict['embedds']))
                
        # compute the affinity matrix
        A = self.get_A_fn[self.AM_function](concat_embedds, to_cpu=True)
    
        initial_A = torch.clone(A)
               
        if self.AM_threshold_strategy != None:
            # remove weak connections with the choosen threshold strategy and value
            logger.info(f' Affinity Matrix Threshold to be used: {self.AM_threshold_strategy} -> {\
                self.get_A_treshold(A) if self.AM_threshold_strategy == 'mean' else self.AM_threshold}')
                
            if self.AM_function != 'rbfk': A = torch.where(A < self.get_A_treshold(A), 0, A)
            else: A = torch.where(A > self.get_A_treshold(A), 1, A)
        
        if self.AM_strategy == 'diversity':
            # set the whole matrix as a distance matrix and not similarity matrix
            A = 1 - A
        elif self.AM_strategy == 'mixed':    
            # set the unlabelled submatrix as distance matrix and not similarity matrix
            n_lab_obs = len(self.labelled_indices)
            
            if self.AM_function == 'rbfk':
                A[:n_lab_obs, :n_lab_obs] = 1 - A[:n_lab_obs, :n_lab_obs] #LL to similarity
                
            else:
                A[n_lab_obs:, n_lab_obs:] = 1 - A[n_lab_obs:, n_lab_obs:] #UU to distance
                
                A[:n_lab_obs, :n_lab_obs] = 1 - A[:n_lab_obs, :n_lab_obs] #UL to distance
                A[n_lab_obs:, n_lab_obs:] = 1 - A[n_lab_obs:, n_lab_obs:] #LU to distance

                
        # plot the TSNE fo the original and modified affinity matrix
        logger.info('Plotting TSNE of the original and modified Affinity Matrix...')
        plot_tsne_A(
            (initial_A, A),
            (self.lab_embedds_dict['labels'], self.unlab_embedds_dict['labels']), self.dataset.classes,
            self.ct_p['timestamp'], self.ct_p['dataset_name'], self.ct_p['trial'], self.strategy_name, self.AM_function, self.AM_strategy, self.iter
        )
        
        self.A = A.to(self.device)
        
        del A
        del initial_A
        del concat_embedds
        gc.collect()
        logger.info(' DONE\n')


    def get_A_rbfk(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        
        device = torch.device('cpu') if to_cpu else self.device
        A_matrix = self.get_A_e_d(concat_embedds, to_cpu)
        seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1)[:, 6]]            
            
        sigmas = torch.unsqueeze(torch.tensor([
            self.get_A_fn[self.AM_function](torch.cat((
                torch.unsqueeze(concat_embedds[i], dim=0), torch.unsqueeze(seventh_neigh[i], dim=0)
            )), to_cpu=to_cpu)[0,1].item()
            for i in range(concat_embedds.shape[0]) 
        ], device=device), dim=0)
        
        A = torch.exp(-A_matrix.pow(2) / (torch.mm(sigmas.T, sigmas))).to(device)
        A = torch.clamp(A, min=0., max=1.)
        
        del A_matrix
        del sigmas
        
        return A
    
    
    def get_A_e_d(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        A = torch.cdist(concat_embedds, concat_embedds).to(torch.device('cpu') if to_cpu else self.device)
        return torch.clamp(A, min=0., max=1.)


    def get_A_cos_sim(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        device = torch.device('cpu') if to_cpu else self.device
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(device)
        
        A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(device)
        A.fill_diagonal_(1.)
        A = torch.clamp(A, min=0., max=1.)

        del normalized_embedding
        return A
        
        
    def get_A_corr(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        A = torch.corrcoef(concat_embedds).to(torch.device('cpu') if to_cpu else self.device)
        return torch.clamp(A, min=0., max=1.)
        

    def get_X(self) -> None:
        logger.info(' => Computing X Matrix...')
        
        self.X: torch.Tensor = torch.zeros(
            (len(self.labelled_indices) + self.len_unlab_sample, self.dataset.n_classes),
            dtype=torch.float32, device=self.device
        )
        for idx, label in enumerate(self.lab_embedds_dict['labels']): self.X[idx][int(label.item())] = 1.
        for idx in range(len(self.labelled_indices), len(self.labelled_indices) + self.len_unlab_sample):
            for label in range(self.dataset.n_classes): self.X[idx][label] = 1. / self.dataset.n_classes
        
        del self.lab_embedds_dict
        logger.info(' DONE\n')
        
        
        
    def graph_trasduction_game(self) -> None:
        
        self.get_A()
        self.get_X()
                
        # from here to the end will be all on cuda
        
        err = float('Inf')
        i = 0
        
        logger.info(' => Running GTG')
        
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = torch.clone(self.X)
            
            self.X *= torch.mm(self.A, self.X)
            self.X /= torch.sum(self.X, dim=1, keepdim=True)            
        
            iter_entropy = entropy(self.X) # there are both labelled and sample unlabelled
            # I have to map only the sample_unlabelled to the correct position
        
            self.unlab_entropy_hist[:, i] = iter_entropy[len(self.labelled_indices):]
                        
            err = torch.norm(self.X - X_old)
            i += 1
        
            '''del X_old
            del iter_entropy
            logger.info(gc.collect())
            torch.cuda.empty_cache()'''
            
        logger.info(' DONE\n')


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:            
        # set the entire batch size to the dimension of the sampled unlabelled set
        self.len_unlab_sample = len(sample_unlab_subset)

        dl_dict = dict(batch_size=self.batch_size, shuffle=False, pin_memory=True)
        
        # we have the batch size which is equal to the number of sampled observation from the unlabelled set
        # set shuffle to false since I do not have interest on shufflind the dataloader, since I have only to get the embeddings
        # thus there is no needs on shuffling the unlabelled dataloader            
        
        unlab_train_dl = DataLoader(sample_unlab_subset, **dl_dict)
        lab_train_dl = DataLoader(Subset(self.dataset.train_ds, self.labelled_indices), **dl_dict)
                        
        logger.info(' => Getting the labelled and unlabelled embeddings')
        self.lab_embedds_dict = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=torch.device('cpu')),
            'labels': torch.empty(0, dtype=torch.int8, device=torch.device('cpu'))
        }
        self.unlab_embedds_dict = {
            'idxs': torch.empty(0, dtype=torch.int8, device=torch.device('cpu')),
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=torch.device('cpu')),
            'labels': torch.empty(0, dtype=torch.int8, device=torch.device('cpu'))
        }        
        self.load_best_checkpoint()
                
        self.get_embeddings(lab_train_dl, self.lab_embedds_dict, embedds2cpu=True)
        self.get_embeddings(unlab_train_dl, self.unlab_embedds_dict, embedds2cpu=True)
        logger.info(' DONE\n')
        
        
        # I save the entropy history in order to be able to plot it
        self.unlab_entropy_hist = torch.zeros((self.len_unlab_sample, self.gtg_max_iter), device=self.device)
                
        logger.info(' => Execution of the Graph Trasduction Game')
        self.graph_trasduction_game()
        logger.info(' DONE\n')
                        
        # TRUE / FALSE np.ndarray
        self.gtg_result_prediction = (torch.argmax(self.X[len(self.labelled_indices):], dim=1).cpu() == self.unlab_embedds_dict['labels']).numpy()

        
        del self.A
        del self.X
        gc.collect()
        torch.cuda.empty_cache()
        
                            
        logger.info(f' => Extracting the Top-k unlabelled observations using {self.ent_strategy}')
        
        if self.ent_strategy == 'mean':
            # computing the mean of the entropis history
            mean_ent = torch.mean(self.unlab_entropy_hist, dim=1)
            overall_topk = torch.topk(mean_ent, k=n_top_k_obs).indices.tolist()
          
        elif self.ent_strategy == 'integral':
            # computing the area of the entropis history using trapezoid formula 
            area = torch.trapezoid(self.unlab_entropy_hist, dim=1)
            overall_topk = torch.topk(area, k=n_top_k_obs, largest=True).indices.tolist()
        
        else: 
            logger.exception('Unrecognized derivates computation strategy')
            raise Exception('Unrecognized derivates computation strategy')
        
        # plot entropy hisstory tensor
        plot_gtg_entropy_tensor(
            tensor=self.unlab_entropy_hist, topk=overall_topk, lab_unlabels=self.unlab_embedds_dict['labels'].tolist(), classes=self.dataset.classes, 
            path=self.path, iter=self.iter - 1, max_x=self.gtg_max_iter, dir='history'
        )
                    
        del self.unlab_entropy_hist
        gc.collect()
        torch.cuda.empty_cache() 
        
        logger.info(' DONE\n')
        
        return overall_topk, [int(self.unlab_embedds_dict['idxs'][id].item()) for id in overall_topk]
    