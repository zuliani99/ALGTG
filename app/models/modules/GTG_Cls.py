
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import entropy, log_assert

from typing import Any, List, Tuple, Dict

import logging
logger = logging.getLogger(__name__)


class Custom_GAP_Module(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(Custom_GAP_Module, self).__init__()

        # same parameters of loss net
        feature_sizes = params['feature_sizes']
        num_channels = params['num_channels']
        interm_dim = params['interm_dim']

        self.convs, self.linears, self.gaps = [], [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.convs.append(nn.Sequential(
                nn.Conv2d(n_c, n_c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(n_c),
                nn.ReLU(),
            ))
            self.gaps.append(nn.AvgPool2d(e_d))
            self.linears.append(nn.Sequential(
                nn.Linear(n_c, interm_dim),
                nn.ReLU(),
            ))

        self.convs = nn.ModuleList(self.convs)
        self.linears = nn.ModuleList(self.linears)
        self.gaps = nn.ModuleList(self.gaps)

        self.linear = nn.Sequential(nn.Linear(interm_dim * len(num_channels), 1))


    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.convs[i](features[i])
            out = self.gaps[i](out)
            out = out.view(out.size(0), -1)
            out = self.linears[i](out)
            outs.append(out)

        out = self.linear(torch.cat(outs, 1))
        return out
    

# can overfit a batch
class Custom_MLP(nn.Module):
    def __init__(self, in_dim):
        super(Custom_MLP, self).__init__()

        self.linear1 = nn.Sequential(nn.Linear(in_dim, in_dim//2), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(in_dim//2, in_dim//4), nn.ReLU())
        self.linear3 = nn.Sequential(nn.Linear(in_dim//4, in_dim//8), nn.ReLU())
        self.classifier = nn.Linear(in_dim//8, 1)
            

    def forward(self, x): 
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.classifier(out)
        return out
    
    


class GTG_Module(nn.Module):
    def __init__(self, params):
        super(GTG_Module, self).__init__()
        
        self.name = self.__class__.__name__

        gtg_p, ll_p = params

        self.get_A_fn = {
            'cos_sim': self.get_A_cos_sim,
            'corr': self.get_A_corr,
            'rbfk': self.get_A_rbfk,
        }
        
        self.gtg_tol: float = gtg_p['gtg_t']
        self.gtg_max_iter: int = gtg_p['gtg_i']
        
        self.AM_strategy: str = gtg_p['am_s']
        self.AM_threshold_strategy: str = gtg_p['am_ts']
        self.AM_threshold: float = gtg_p['am_t']
        
        self.ent_strategy: str = gtg_p['e_s']
        self.perc_labeled_batch: int = gtg_p['plb']
        
        self.n_top_k_obs: int = gtg_p['n_top_k_obs']
        self.n_classes: int = gtg_p['n_classes']
        self.device: int = gtg_p['device']

        #self.c_mod = Custom_GAP_Module(ll_p).to(self.device)
        self.c_mod = Custom_MLP(ll_p['num_channels'][-1]).to(self.device)
        
        
        
    def define_A_function(self, AM_function: str) -> None: self.AM_function: str = AM_function
    
    
    def get_A_treshold(self, A: torch.Tensor) -> Any:
        if self.AM_threshold_strategy == 'mean': return torch.mean(A)
        else: return self.AM_threshold
    
    
    def get_A_rbfk(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A_matrix = self.get_A_e_d(concat_embedds)
        seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1)[:, 6]]            

        sigmas = torch.unsqueeze(torch.tensor([
            self.get_A_e_d(torch.cat((
                torch.unsqueeze(concat_embedds[i], dim=0), torch.unsqueeze(seventh_neigh[i], dim=0)
            )))[0,1].item()
            for i in range(concat_embedds.shape[0]) 
        ], device=self.device), dim=0)
        
        '''A = torch.exp(-A_matrix.pow(2) / (torch.mm(sigmas.T, sigmas))).to(self.device)      
        
        return torch.clamp(A, min=0., max=1.)'''
        
        A_m_2 = -A_matrix.pow(2)
        sigma_T = sigmas.T
        sigma_mm = (torch.mm(sigma_T, sigmas))
        A = torch.exp(A_m_2 / sigma_mm)
        return torch.clamp(A.to(self.device), min=0., max=1.)
    
        
    def get_A_e_d(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A = torch.cdist(concat_embedds, concat_embedds).to(self.device)
        return torch.clamp(A, min=0.1, max=1.)
    

    def get_A_cos_sim(self, concat_embedds: torch.Tensor) -> torch.Tensor: 
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(self.device)
        A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device)
        
        '''A.fill_diagonal_(1.)
        
        return torch.clamp(A, min=0., max=1.)'''
        A_flat = A.flatten()
        A_flat[::(A.shape[-1]+1)] = 0.
        return torch.clamp(A.clone(), min=0., max=1.)

        
    def get_A_corr(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A = torch.corrcoef(concat_embedds).to(self.device)
        return torch.clamp(A, min=0., max=1.)
    
    
    def get_A(self, embedding: torch.Tensor) -> None:
        # compute the affinity matrix
        if self.AM_function != None: A = self.get_A_fn[self.AM_function](embedding)
        else: raise AttributeError('A Fucntion is None')
                
        if self.AM_threshold_strategy != None and self.AM_threshold!= None:
            if self.AM_function != 'rbfk': A = torch.where(A < self.get_A_treshold(A), 0, A)
            else: A = torch.where(A > self.get_A_treshold(A), 1, A)

        if self.AM_strategy == 'diversity':
            # set the whole matrix as a distance matrix and not similarity matrix
            A = 1 - A
        elif self.AM_strategy == 'mixed':    
            # set the unlabeled submatrix as distance matrix and not similarity matrix
            
            if self.AM_function == 'rbfk':
                for i in self.labeled_indices:  # -> insert the similarity for the labeled observations
                    for j in self.labeled_indices:
                        A[i,j] = 1 - A[i,j]
                        A[j,i] = 1 - A[j,i]
            else:
                A = 1 - A # -> all distance matrix
                for i in self.labeled_indices:  # -> reinsert the similarity for the labeled observations
                    for j in self.labeled_indices:
                        A[i,j] = 1 - A[i,j]
                        A[j,i] = 1 - A[j,i]                        
        
        self.A = A
        

    def get_X(self) -> None:
        self.X: torch.Tensor = torch.zeros((self.batch_size, self.n_classes), dtype=torch.float32, device=self.device)
        for idx, label in zip(self.labeled_indices, self.lab_labels):
            self.X[idx][int(label.item())] = 1.
        for idx in self.unlabeled_indices:
            for label in range(self.n_classes): self.X[idx][label] = 1. / self.n_classes



    @torch.no_grad()
    def graph_trasduction_game_detached(self, embedding: torch.Tensor) -> torch.Tensor:
        
        entropy_hist = torch.zeros((self.batch_size, self.gtg_max_iter), device=self.device)        
        
        self.get_A(embedding)
        self.get_X()
                
        err = float('Inf')
        i = 0
                
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = torch.clone(self.X)
            
            self.X *= torch.mm(self.A, self.X)
            
            self.X /= torch.sum(self.X, dim=1, keepdim=True)
            
            iter_entropy = entropy(self.X).to(self.device)
            
            entropy_hist[self.unlabeled_indices, i] = iter_entropy[self.unlabeled_indices]

            i += 1
            err = torch.norm(self.X - X_old)

        return entropy_hist



    def graph_trasduction_game(self, embedding: torch.Tensor) -> torch.Tensor:
        
        entropy_hist = torch.zeros((self.batch_size, self.gtg_max_iter), device=self.device)#, requires_grad=True)        
        
        self.get_A(embedding)
        self.get_X()
        self.X.requires_grad_(True)
        
        err = float('Inf')
        i = 0
                
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = self.X.detach().clone()
            
            mm_A_X = torch.mm(self.A, self.X)
            mult_X_A_X = self.X * mm_A_X
            sum_X_A_X = torch.sum(mult_X_A_X, dim=1, keepdim=True)
            div_sum_X_A_X = mult_X_A_X / sum_X_A_X
            
            entropy_hist[:, i] = entropy(div_sum_X_A_X).to(self.device)
            
            err = torch.norm(div_sum_X_A_X.detach() - X_old)            
            self.X = div_sum_X_A_X
            
            i += 1
                    
        return entropy_hist.requires_grad_(True)
    
        
        
    # List[torch.Tensor] -> detection
    # , mode: int
    def preprocess_inputs(self, embedding: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        indices = torch.arange(self.batch_size)
        #indices = torch.randperm(self.batch_size)

        '''if mode == 1:
            self.labeled_indices: List[int] = indices[:self.n_lab_obs].tolist()
            self.unlabeled_indices: List[int] = indices[self.n_lab_obs:].tolist()
        else:
            self.unlabeled_indices: List[int] = indices[:self.n_lab_obs].tolist()
            self.labeled_indices: List[int] = indices[self.n_lab_obs:].tolist()'''
        
        self.labeled_indices: List[int] = indices[:self.n_lab_obs].tolist()
        self.unlabeled_indices: List[int] = indices[self.n_lab_obs:].tolist()
                                
        labeled_mask = torch.zeros(self.batch_size, device=self.device)
        labeled_mask[self.labeled_indices] = 1.
        
        self.lab_labels = labels[self.labeled_indices]
        self.unlab_labels = labels[self.unlabeled_indices]
                
        entropy_hist = self.graph_trasduction_game_detached(embedding)
        #entropy_hist = self.graph_trasduction_game(embedding)
        
        if self.ent_strategy == 'mean':
            # computing the mean of the entropis history
            quantity_result = torch.mean(entropy_hist, dim=1)
            
        elif self.ent_strategy == 'integral':
            # computing the area of the entropis history using trapezoid formula 
            quantity_result = torch.trapezoid(entropy_hist, dim=1)
            
        else:
            logger.exception(' Invlaid GTG Strategy') 
            raise AttributeError(' Invlaid GTG Strategy')
            
        return quantity_result, labeled_mask
    
    
    
    def forward(self, features: List[torch.Tensor], embedds: torch.Tensor, labels: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    # Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    # Tuple[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
    
        #y_pred = self.c_mod(features, embedds).squeeze()
        y_pred = self.c_mod(embedds).squeeze()
        #y_pred = self.c_mod(features).squeeze()
        
        self.batch_size = len(embedds)
        self.n_lab_obs = int(self.batch_size * self.perc_labeled_batch) 
        
        y_true, labeled_mask = self.preprocess_inputs(embedds, labels)
        #y_true_1, labeled_mask_1 = self.preprocess_inputs(embedds, labels, mode=1)
        #y_true_2, labeled_mask_2 = self.preprocess_inputs(embedds, labels, mode=2)
        #logger.info(f'y_pred {y_pred}\ny_true {y_true}')
            
        #return (y_pred, (y_true_1, y_true_2)), (labeled_mask_1.bool(), labeled_mask_2.bool())
        return (y_pred, y_true), labeled_mask.bool()