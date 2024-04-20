
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights_apply
from utils import Entropy_Strategy, entropy

from typing import Any, List, Dict, Tuple


import logging
logger = logging.getLogger(__name__)


'''class Custom_Module(nn.Module):
    def __init__(self, embedding_dim: int):
        super(Custom_Module, self).__init__()
        
        self.layers = []
        dim_ll = embedding_dim
        for i in range(5):
            self.layers.append(nn.Linear(in_features = dim_ll, out_features = embedding_dim // (2**(i+1))))
            self.layers.append(nn.BatchNorm1d(num_features = embedding_dim // (2**(i+1))))
            self.layers.append(nn.ReLU())
            dim_ll = embedding_dim // (2**(i+1))
        self.layers.append(nn.Linear(in_features = dim_ll, out_features = 1))
        self.layers = nn.ModuleList(self.layers)
        
        self.apply(init_weights_apply)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers: x = layer(x)
        return x'''
        
class Custom_Module(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(Custom_Module, self).__init__()
        
        # same parameters of loss net
        feature_sizes = params['feature_sizes']
        num_channels = params['num_channels']
        interm_dim = params['interm_dim']

        self.module_list, self.sequentials_1, self.sequentials_2 = [], [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.sequentials_1.append(nn.Sequential(
                nn.Conv2d(n_c, n_c // (e_d // 2), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(n_c // (e_d // 2)),
            ))
            self.sequentials_2.append(nn.Sequential(
                nn.Linear(n_c * (e_d // 2), interm_dim),
                nn.ReLU(),
                #nn.BatchNorm1d(interm_dim)
            )),

        self.sequentials_1 = nn.ModuleList(self.sequentials_1)
        self.sequentials_2 = nn.ModuleList(self.sequentials_2)
        self.linear = nn.Linear(interm_dim * len(num_channels), 1)

        self.apply(init_weights_apply)


    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.sequentials_1[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.sequentials_2[i](out)
            outs.append(out)

        out = F.relu(self.linear(torch.cat(outs, 1)))
        return out



class GTG_Module(nn.Module):
    def __init__(self, params, phase='train'):
        super(GTG_Module, self).__init__()

        gtg_p, ll_p = params

        self.get_A_fn = {
            'cos_sim': self.get_A_cos_sim,
            'corr': self.get_A_corr,
            'e_d': self.get_A_e_d,
        }
        
        self.gtg_tol: float = gtg_p['gtg_tol']
        self.gtg_max_iter: int = gtg_p['gtg_max_iter']
        self.strategy_type: str = gtg_p['strategy_type']
        self.perc_labeled_batch: int = gtg_p['perc_labeled_batch']
        
        self.n_top_k_obs: int = gtg_p['n_top_k_obs']
        self.n_classes: int = gtg_p['n_classes']
        self.device: int = gtg_p['device']
        self.phase: str = phase
        
        self.c_cnn = Custom_Module(ll_p).to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)
        self.l1loss = nn.L1Loss().to(self.device)
    
    
    
    def define_additional_parameters(self, remaining_param: Dict[str, Any]) -> None:
        self.rbf_aff: bool = remaining_param['rbf_aff']
        self.A_function: str = remaining_param['A_function']
        self.ent_strategy: Entropy_Strategy = remaining_param['ent_strategy']
        self.threshold_strategy: str = remaining_param['threshold_strategy']
        self.threshold: float = remaining_param['threshold']
        
        
    def change_pahse(self, new_phase: str) -> None:
        self.phase = new_phase
        
    
    def get_A_treshold(self, A: torch.Tensor) -> Any:
        if self.threshold_strategy == 'mean': return torch.mean(A)
        else: return self.threshold
    
    
    def get_A_rbfk(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        
        A_matrix = self.get_A_fn[self.A_function](concat_embedds) # type: ignore

        if self.A_function == 'e_d':
            # if euclidean distance is choosen we take the 7th smallest observation which is the 7th closest one (ascending order)
            seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1)[:, 6]]            
        else:
            # if correlation or cosine_similairty are choosen we take the 7th highest observation which is the 7th most similar one (descending order)
            seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1, descending=True)[:, 6]]
            
        sigmas = torch.unsqueeze(torch.tensor([
            self.get_A_fn[self.A_function](torch.cat(( # type: ignore
                torch.unsqueeze(concat_embedds[i], dim=0), torch.unsqueeze(seventh_neigh[i], dim=0)
            )))[0,1].item()
            for i in range(concat_embedds.shape[0]) 
        ], device=self.device), dim=0)
        
        A_m_2 = -A_matrix.pow(2)
        sigma_T = sigmas.T
        sigma_mm = (torch.mm(sigma_T, sigmas))
        A = torch.exp( A_m_2 / sigma_mm) 
        return A.to(self.device)
    
        
    def get_A_e_d(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A = torch.cdist(concat_embedds, concat_embedds).to(self.device)
        return A
    

    def get_A_cos_sim(self, concat_embedds: torch.Tensor) -> torch.Tensor:        
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(self.device)
        A = F.relu(torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device))
        A.fill_diagonal_(1.)
        return A
        
        
    def get_A_corr(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A = F.relu(torch.corrcoef(concat_embedds).to(self.device))
        return A
    
    
    def get_A(self, embedding: torch.Tensor) -> None:

        # compute the affinity matrix
        if self.A_function != None:
            A = self.get_A_rbfk(embedding) if self.rbf_aff else self.get_A_fn[self.A_function](embedding)
        else: raise AttributeError('A Fucntion is None')

        if self.threshold_strategy != None and self.threshold != None:
            #logger.info(f' Affinity Matrix Threshold to be used: {self.threshold_strategy}, {self.threshold} -> {self.get_A_treshold(A)}')
            if self.A_function != 'e_d': A = torch.where(A < self.get_A_treshold(A), 0, A)
            else: A = torch.where(A > self.get_A_treshold(A), 1, A)

        if self.strategy_type == 'diversity':
            # set the whole matrix as a distance matrix and not similarity matrix
            A = 1 - A
        elif self.strategy_type == 'mixed':    
            # set the unlabeled submatrix as distance matrix and not similarity matrix
            
            if self.A_function == 'e_d':
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
        #self.A.requires_grad_(True)
        
        

    def get_X(self) -> None:
        self.X: torch.Tensor = torch.zeros((self.batch_size, self.n_classes), dtype=torch.float32, device=self.device)
        for idx, label in zip(self.labeled_indices, self.lab_labels):
            self.X[idx][int(label.item())] = 1.
        for idx in self.unlabeled_indices:
            for label in range(self.n_classes): self.X[idx][label] = 1. / self.n_classes
        
        #self.X.requires_grad_(True)
                 
                 
    @torch.no_grad()
    def check_increasing_sum(self, mult_X_A_X: torch.Tensor, old_rowsum_X: int) -> int:
    #def check_increasing_sum(self, old_rowsum_X: int) -> int:
        #rowsum_X = torch.sum(self.X).item()
        rowsum_X = torch.sum(mult_X_A_X).item()
        if rowsum_X < old_rowsum_X: # it has to be increasing or at least equal
            logger.exception('Sum of the vector on the denominator is lower than the previous step')
            raise Exception('Sum of the vector on the denominator is lower than the previous step')
        return int(rowsum_X)

        
    '''@torch.no_grad()
    def graph_trasduction_game(self, embedding: torch.Tensor) -> None:
        
        self.get_A(embedding)
        self.get_X()

        assert torch.all(self.X >= 0), 'Negative values in self.X'
        if not torch.all(self.A >= 0): print(embedding, self.A)
        assert torch.all(self.A >= 0), 'Negative values in self.A'

        err = float('Inf')
        i = 0
        old_rowsum_X = 0
        
        
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = torch.clone(self.X)
            
            A_X = torch.mm(self.A, self.X)
            assert torch.all(A_X >= 0), 'Negative values in A_X'
            
            self.X = self.X * A_X
            assert torch.all(self.X >= 0), 'Negative values in self.X'
            old_rowsum_X = self.check_increasing_sum(old_rowsum_X)
            
            X_sum = torch.sum(self.X, dim=1, keepdim=True)
            assert torch.all(X_sum > 0), 'Negative or zero values in X_sum'
            
            self.X = self.X / X_sum 
            assert torch.all(self.X >= 0), 'Negative values in self.X'
            
            iter_entropy = entropy(self.X).to(self.device) # there are both labeled and sample unlabeled
            assert torch.all(iter_entropy >= 0), 'Negative values in iter_entropy'

            self.unlab_entropy_hist[:, i] = iter_entropy
            
            i += 1
            err = torch.norm(self.X - X_old)'''
            
            
    def graph_trasduction_game(self, embedding) -> None:
        
        self.get_A(embedding)
        #self.A.register_hook(lambda t: print(f'hook self.A :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
        self.get_X()
        
        assert torch.all(self.A >= 0), 'Negative value in self.A'
        assert torch.all(self.X >= 0), 'Negative value in self.X'
        
        err = float('Inf')
        i = 0
        old_rowsum_X = 0
        
        unlab_entropy_hist_while = torch.clone(self.unlab_entropy_hist)
        
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = torch.clone(self.X)
            assert torch.all(X_old >= 0), 'Negative values in X_old'
            
            mm_A_X = torch.matmul(self.A, self.X)
            #mm_A_X.register_hook(lambda t: print(f'hook mm_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            assert torch.all(mm_A_X >= 0), 'Negative values in mm_A_X'
            
            mult_X_A_X = self.X * mm_A_X #torch.mul(self.X, mm_A_X)
            #mult_X_A_X.register_hook(lambda t: print(f'hook mult_X_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            assert torch.all(mult_X_A_X >= 0), 'Negative values in mult_X_A_X'
                        
            old_rowsum_X = self.check_increasing_sum(mult_X_A_X, old_rowsum_X)
            
            mult_X_A_X = X_old + mult_X_A_X
            
            sum_X_A_X = torch.sum(mult_X_A_X, dim=1, keepdim=True) + 1e-8
            #sum_X_A_X.register_hook(lambda t: print(f'hook sum_X_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            assert torch.all(sum_X_A_X > 0), 'Negative or zero values in sum_X_A_X'
            
            div_sum_X_A_X = mult_X_A_X / sum_X_A_X
            #div_sum_X_A_X.register_hook(lambda t: print(f'hook div_sum_X_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            assert torch.all(div_sum_X_A_X >= 0), 'Nagative values in div_sum_X_A_X'    
            
            
            iter_entropy = entropy(div_sum_X_A_X).to(self.device)
            assert torch.all(iter_entropy >= 0), 'Nagative values in iter_entropy'
            
            #div_sum_X_A_X = X_old + div_sum_X_A_X

            unlab_entropy_hist_while[:, i] = iter_entropy
            
            i += 1
            #err = torch.norm(div_sum_X_A_X.detach() - X_old)            
            err = torch.norm(div_sum_X_A_X - X_old)            
            self.X = div_sum_X_A_X
            
        self.unlab_entropy_hist = unlab_entropy_hist_while
        #self.unlab_entropy_hist.register_hook(lambda t: print(f'hook self.unlab_entropy_hist :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
        assert torch.all(self.unlab_entropy_hist >= 0), 'Nagative values in self.unlab_entropy_hist'
        
        
        
    # List[torch.Tensor] -> detection
    def preprocess_inputs(self, embedding: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        self.batch_size = len(embedding)
        self.n_lab_obs = int(self.batch_size * self.perc_labeled_batch) 
        
        shuffled_indices = torch.randperm(self.batch_size)

        self.labeled_indices: List[int] = shuffled_indices[:self.n_lab_obs].tolist()
        self.unlabeled_indices: List[int] = shuffled_indices[self.n_lab_obs:].tolist()
                        
        self.unlab_entropy_hist = torch.zeros((self.batch_size, self.gtg_max_iter), device=self.device, requires_grad=True)        
        
        mask = torch.zeros(self.batch_size, device=self.device, requires_grad=False)
        mask[self.labeled_indices] = 1.
        
        self.lab_labels = labels[self.labeled_indices]
        self.unlab_labels = labels[self.unlabeled_indices]
                
        self.graph_trasduction_game(embedding)
        
        if self.ent_strategy is Entropy_Strategy.MEAN:
            # computing the mean of the entropis history
            quantity_result = torch.mean(self.unlab_entropy_hist, dim=1)
            #quantity_result.register_hook(lambda t: print(f'hook quantity_result :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            
        elif self.ent_strategy is Entropy_Strategy.H_INT:
            # computing the area of the entropis history using trapezoid formula 
            quantity_result = torch.trapezoid(self.unlab_entropy_hist, dim=1)
            #quantity_result.register_hook(lambda t: print(f'hook quantity_result :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            
        else:
            logger.exception(' Invlaid GTG Strategy') 
            raise AttributeError(' Invlaid GTG Strategy')
            
        return quantity_result, mask
    
    
    
    def forward(self, features: List[torch.Tensor], embedds: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        #for feature in features: print(feature.grad, feature.grad_fn)
        y_pred = self.c_cnn(features).squeeze()
        #print('y_pred', y_pred)
        #y_pred.register_hook(lambda t: print(f'hook y_pred :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
        
        if self.phase == 'train':
            
            y_true, mask = self.preprocess_inputs(embedds, labels) # type: ignore
            #logger.info(f'{y_pred} - {y_true}')
            return self.mse_loss(y_pred, y_true), mask if mask == None else mask.bool()
            #return self.l1loss(y_pred, y_true), mask if mask == None else mask.bool()
        else:
            return y_pred, None        
        
