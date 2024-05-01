
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import entropy, log_assert

from typing import Any, List, Tuple

import logging
logger = logging.getLogger(__name__)

        

class Custom_GAP_Module(nn.Module):
    def __init__(self, params):
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


class GTG_Module(nn.Module):
    def __init__(self, params):
        super(GTG_Module, self).__init__()
        
        self.name = self.__class__.__name__

        gtg_p, ll_p = params

        self.get_A_fn = {
            'cos_sim': self.get_A_cos_sim,
            'corr': self.get_A_corr,
            'e_d': self.get_A_e_d,
        }
        
        self.gtg_tol: float = gtg_p['gtg_t']
        self.gtg_max_iter: int = gtg_p['gtg_i']
        
        self.AM_function: str = gtg_p['am']
        self.AM_strategy: str = gtg_p['am_s']
        self.AM_threshold_strategy: str = gtg_p['am_ts']
        self.AM_threshold: float = gtg_p['am_t']
        
        self.ent_strategy: str = gtg_p['e_s']
        self.rbf_aff: bool = gtg_p['rbfk']
                
        self.perc_labeled_batch: int = gtg_p['plb']
        
        
        self.n_top_k_obs: int = gtg_p['n_top_k_obs']
        self.n_classes: int = gtg_p['n_classes']
        self.device: int = gtg_p['device']

        self.c_mod = Custom_GAP_Module(ll_p).to(self.device)
        
        
    
    def get_A_treshold(self, A: torch.Tensor) -> Any:
        if self.AM_threshold_strategy == 'mean': return torch.mean(A)
        else: return self.AM_threshold
    
    
    def get_A_rbfk(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        
        A_matrix = self.get_A_fn[self.AM_function](concat_embedds) # type: ignore

        if self.AM_function == 'e_d':
            # if euclidean distance is choosen we take the 7th smallest observation which is the 7th closest one (ascending order)
            seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1)[:, 6]]            
        else:
            # if correlation or cosine_similairty are choosen we take the 7th highest observation which is the 7th most similar one (descending order)
            seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1, descending=True)[:, 6]]
            
        sigmas = torch.unsqueeze(torch.tensor([
            self.get_A_fn[self.AM_function](torch.cat(( # type: ignore
                torch.unsqueeze(concat_embedds[i], dim=0), torch.unsqueeze(seventh_neigh[i], dim=0)
            )))[0,1].item()
            for i in range(concat_embedds.shape[0]) 
        ], device=self.device), dim=0)
        
        A = torch.exp(-A_matrix.pow(2) / (torch.mm(sigmas.T, sigmas))).to(self.device)      
        
        '''A_m_2 = -A_matrix.pow(2)
        sigma_T = sigmas.T
        sigma_mm = (torch.mm(sigma_T, sigmas))
        A = torch.exp(A_m_2 / sigma_mm) '''
        #return torch.clamp(A.to(self.device), min=0., max=1.)
        return torch.clamp(A, min=0., max=1.)
    
        
    def get_A_e_d(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A = torch.cdist(concat_embedds, concat_embedds).to(self.device)
        return torch.clamp(A, min=0.1, max=1.)
    

    def get_A_cos_sim(self, concat_embedds: torch.Tensor) -> torch.Tensor: 
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(self.device)
        A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device)
        
        A.fill_diagonal_(1.)
        
        '''A_flat = A.flatten()
        A_flat[::(A.shape[-1]+1)] = 0.
        return torch.clamp(A.clone(), min=0., max=1.)'''
        return torch.clamp(A, min=0., max=1.)

        
    def get_A_corr(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A = torch.corrcoef(concat_embedds).to(self.device)
        return torch.clamp(A, min=0., max=1.)
    
    
    def get_A(self, embedding: torch.Tensor) -> None:

        # compute the affinity matrix
        if self.AM_function != None:
            A = self.get_A_rbfk(embedding) if self.rbf_aff else self.get_A_fn[self.AM_function](embedding)
        else: raise AttributeError('A Fucntion is None')
                
        if self.AM_threshold_strategy != None and self.AM_threshold!= None:
            if self.AM_function != 'e_d': A = torch.where(A < self.get_A_treshold(A), 0, A)
            else: A = torch.where(A > self.get_A_treshold(A), 1, A)

        if self.AM_strategy == 'diversity':
            # set the whole matrix as a distance matrix and not similarity matrix
            A = 1 - A
        elif self.AM_strategy == 'mixed':    
            # set the unlabeled submatrix as distance matrix and not similarity matrix
            
            if self.AM_function == 'e_d':
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
            entropy_hist[:, i] = entropy(self.X).to(self.device)
            
            i += 1
            err = torch.norm(self.X - X_old)            

        return entropy_hist
    
            
    def graph_trasduction_game(self, embedding: torch.Tensor) -> torch.Tensor:
        
        entropy_hist = torch.zeros((self.batch_size, self.gtg_max_iter), device=self.device, requires_grad=True)        
        
        self.get_A(embedding)
        #self.A.register_hook(lambda t: print(f'hook self.A :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
        self.get_X()
        self.X.requires_grad_(True)
        
                
        err = float('Inf')
        i = 0
        
        entropy_hist_while = torch.clone(entropy_hist)
        
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = torch.clone(self.X)
            log_assert(torch.all(X_old >= 0), 'Negative values in X_old')
            
            mm_A_X = torch.mm(self.A, self.X)
            #mm_A_X.register_hook(lambda t: print(f'hook mm_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            log_assert(torch.all(mm_A_X >= 0), 'Negative values in mm_A_X')
            
            mult_X_A_X = self.X * mm_A_X
            #mult_X_A_X.register_hook(lambda t: print(f'hook mult_X_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            log_assert(torch.all(mult_X_A_X >= 0), 'Negative values in mult_X_A_X')
                        
            
            sum_X_A_X = torch.sum(mult_X_A_X, dim=1, keepdim=True)
            #sum_X_A_X.register_hook(lambda t: print(f'hook sum_X_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            log_assert(torch.all(sum_X_A_X > 0), 'Negative or zero values in sum_X_A_X')
            
            div_sum_X_A_X = mult_X_A_X / sum_X_A_X
            #div_sum_X_A_X.register_hook(lambda t: print(f'hook div_sum_X_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            log_assert(torch.all(div_sum_X_A_X >= 0), 'Nagative values in div_sum_X_A_X')    
            
            
            iter_entropy = entropy(div_sum_X_A_X).to(self.device)
            log_assert(torch.all(iter_entropy >= 0), 'Nagative values in iter_entropy')
            
            entropy_hist_while[:, i] = iter_entropy
            #self.entropy_hist[:, i] = iter_entropy
            
            i += 1
            err = torch.norm(div_sum_X_A_X.detach() - X_old)            
            self.X = div_sum_X_A_X
            
        entropy_hist = entropy_hist_while
        #self.entropy_hist.register_hook(lambda t: print(f'hook self.entropy_hist :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
        log_assert(torch.all(entropy_hist >= 0), 'Nagative values in self.entropy_hist')
        
        #self.entropy_hist.requires_grad_(True)
        return entropy_hist
        
        
        
    # List[torch.Tensor] -> detection
    def preprocess_inputs(self, embedding: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        self.batch_size = len(embedding)
        self.n_lab_obs = int(self.batch_size * self.perc_labeled_batch) 

        indices = torch.arange(self.batch_size)

        self.labeled_indices: List[int] = indices[:self.n_lab_obs].tolist()
        self.unlabeled_indices: List[int] = indices[self.n_lab_obs:].tolist()
                                
        labeled_mask = torch.zeros(self.batch_size, device=self.device)
        labeled_mask[self.labeled_indices] = 1.
        
        self.lab_labels = labels[self.labeled_indices]
        self.unlab_labels = labels[self.unlabeled_indices]
                
        #entropy_hist = self.graph_trasduction_game(embedding)
        entropy_hist = self.graph_trasduction_game_detached(embedding)
        
        
        if self.ent_strategy == 'mean':
            # computing the mean of the entropis history
            quantity_result = torch.mean(entropy_hist, dim=1)
            #quantity_result.register_hook(lambda t: print(f'hook quantity_result :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            
        elif self.ent_strategy == 'integral':
            # computing the area of the entropis history using trapezoid formula 
            quantity_result = torch.trapezoid(entropy_hist, dim=1)
            #quantity_result.register_hook(lambda t: print(f'hook quantity_result :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            
        else:
            logger.exception(' Invlaid GTG Strategy') 
            raise AttributeError(' Invlaid GTG Strategy')
            
        return quantity_result, labeled_mask
    
    
    
    def forward(self, features: List[torch.Tensor], embedds: torch.Tensor, labels: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:

        #y_pred = self.c_mod(features, embedds).squeeze()
        y_pred = self.c_mod(features).squeeze()
        
        y_true, labeled_mask = self.preprocess_inputs(embedds, labels)
        #logger.info(f'y_pred {y_pred[self.unlabeled_indices]}\ny_true {y_true[self.unlabeled_indices]}')
            
        return (y_pred, y_true, self.X), labeled_mask.bool()