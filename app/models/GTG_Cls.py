
import torch
import torch.nn as nn
import torch.nn.functional as F

from ResNet18 import ResNet18
from utils import Entropy_Strategy, entropy

import logging
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple


class GTG_Module(nn.Module):
    def __init__(self, gtg_p, n_top_k_obs, n_classes, init_lab_obs, device, phase='train'):
        super(GTG_Module).__init__()

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
        
        self.phase = phase
        self.n_top_k_obs = n_top_k_obs
        self.n_classes = n_classes
        self.init_lab_obs = init_lab_obs
        
        self.l1 = nn.Linear(512, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 1)
        
        
        self.device = device
        
    def change_pahse(self):
        self.phase = 'test' if self.pahse == 'train' else 'train'
        
    
    def get_A_treshold(self, A: torch.Tensor) -> Any:
        if self.threshold_strategy == 'mean': return torch.mean(A)
        else: return self.threshold
    
    
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
        
                
        del A_matrix
        del sigmas
        torch.cuda.empty_cache()
        
        return A
    
    
        
    def get_A_e_d(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A = torch.cdist(concat_embedds, concat_embedds).to(self.device)
        return A



    def get_A_cos_sim(self, concat_embedds: torch.Tensor) -> torch.Tensor:        
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(self.device)
        
        A = F.relu(torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device))
        A.fill_diagonal_(1.)

        del normalized_embedding
        torch.cuda.empty_cache()
        
        return A
        
        
        
    def get_A_corr(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A = F.relu(torch.corrcoef(concat_embedds).to(self.device))
        return A
    
    
    
    def get_A(self) -> None:

        concat_embedds = torch.cat(
            (self.labeled_obs['embedds'], self.unlabeled_obs['embedds'])
        ).to(self.device)
        
        # compute the affinity matrix
        A = self.get_A_rbfk(concat_embedds, to_cpu=True) if self.rbf_aff else self.get_A_fn[self.A_function](concat_embedds)

        initial_A = torch.clone(A)
        
        # remove weak connections with the choosen threshold strategy and value
        logger.info(f' Affinity Matrix Threshold to be used: {self.threshold_strategy}, {self.threshold} -> {self.get_A_treshold(A)}')
        A = torch.where(
            A < self.get_A_treshold(A) if self.A_function != 'e_d' else A > self.get_A_treshold(A),
            0 if self.A_function != 'e_d' else 1, A
        )
        
        
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
        

        mat_cos_sim = nn.CosineSimilarity(dim=0)
        logger.info(f' Cosine Similarity between the initial matrix and the thresholded one: {mat_cos_sim(initial_A.flatten(), A.flatten()).item()}')

        self.A = A
        
        # plot the TSNE fo the original and modified affinity matrix
        '''plot_tsne_A(
            (initial_A, A),
            (self.lab_embedds_dict['labels'], self.unlabeled_labels), self.dataset.classes,
            self.ct_p['timestamp'], self.ct_p['dataset_name'], self.ct_p['trial'], self.method_name, self.A_function, self.strategy_type, self.iter
        )'''
        
        del A
        del initial_A
        del concat_embedds
        torch.cuda.empty_cache()
        
        

    def get_X(self) -> None:
                
        self.X: torch.Tensor = torch.zeros(
            (self.batch_size, self.n_classes), dtype=torch.float32, device=self.device
        )

        for idx, label in enumerate(self.labeled_obs['labels']): self.X[idx][int(label.item())] = 1.
        
        for idx in range(len(self.init_lab_obs), self.batch_size):
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
        
        del X_old
        torch.cuda.empty_cache()
        
        
    
    def preprocess_inputs(self, embedds, labels):
        self.batch_size = len(embedds)
        
        shuffled_indices = torch.randperm(self.batch_size)

        labeled_indices: List[int] = shuffled_indices[:self.init_lab_obs].tolist()
        unlabeled_indices: List[int] = shuffled_indices[self.init_lab_obs:].tolist()
        
        self.unlab_entropy_hist = torch.zeros((unlabeled_indices, self.gtg_max_iter), device=self.device)
        mask = torch.zeros(self.batch_size, device = self.device)
        mask[labeled_indices] = 1.
        
        self.labeled_obs = dict(
            embedds = embedds[labeled_indices], labels = labels[labeled_indices]
        )
        
        self.unlabeled_obs = dict(
            embedds = embedds[unlabeled_indices], labels = labels[unlabeled_indices]
        )
        
        self.graph_trasduction_game()
        
        del self.A
        del self.X
        del self.labeled_obs
        del self.unlabeled_obs
        torch.cuda.empty_cache()   
        
        if self.ent_strategy is Entropy_Strategy.MEAN:
            # computing the mean of the entropis history
            quantity_result = torch.mean(self.unlab_entropy_hist, dim=1)
            
        elif self.ent_strategy is Entropy_Strategy.H_INT:
            # computing the area of the entropis history using trapezoid formula 
            quantity_result = torch.trapezoid(self.unlab_entropy_hist, dim=1)
        else:
            raise AttributeError(' Invlaid GTG Strategy')
    
        return quantity_result, mask
    
        
    
    def forward(self, embedds, labels):
        
        if self.phase == 'train':
            y_true, mask = self.preprocess_inputs(embedds, labels)
        else:
            y_true, mask = embedds, None
        # for now it takes only as input the embedding of the resnet
        out1 = F.relu(self.l1(y_true))
        out2 = F.relu(self.l2(out1))
        out3 = F.relu(self.l3(out2))
        y_pred = self.l4(out3)
        
        return y_pred, y_true, mask
    
    
    
    
class Class_GTG(nn.Module):
    def __init__(self, gtg_p, n_top_k_obs, init_lab_obs, device, image_size: int, n_classes=10, n_channels=3) -> None:
        super(Class_GTG, self).__init__()
        self.gtg = GTG_Module(gtg_p, n_top_k_obs, n_classes, init_lab_obs, device)
        self.backbone = ResNet18(image_size, n_classes=n_classes, n_channels=n_channels)
        
    def forward(self, images, labels, mode='all'):
        if mode == 'all':
            outs, embedds = self.backbone(images)
            y_pred, y_true, mask = self.gtg(embedds, labels)
            return outs, embedds, nn.MSELoss(y_pred, y_true), mask 
        elif mode == 'probs':
            outs, _ = self.backbone(images)
            return outs
        elif mode == 'embedds':
            _, embedds = self.backbone(images)
            return embedds
        elif mode == 'pred_GTG':
            outs, embedds = self.backbone(images)
            self.gtg(embedds, labels)
        else: 
            raise AttributeError('You have specified wrong output to return for ResNet_LL')
    
