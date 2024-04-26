
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights_apply
from utils import Entropy_Strategy, entropy

from typing import Any, List, Dict, Tuple


import logging
logger = logging.getLogger(__name__)



class Custom_MLP(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(Custom_MLP, self).__init__()

        # same parameters of loss net
        feature_sizes = params['feature_sizes']
        num_channels = params['num_channels']
        interm_dim = params['interm_dim']

        self.module_list, self.sequentials_1, self.sequentials_2 = [], [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.sequentials_1.append(nn.Sequential(
                nn.Linear(n_c * (e_d**2), n_c * (e_d // 2)),
                nn.ReLU(),
                nn.Dropout(),
                nn.BatchNorm1d(n_c * (e_d // 2))

            ))
            self.sequentials_2.append(nn.Sequential(
                nn.Linear(n_c * (e_d // 2), interm_dim),
                nn.ReLU(),
                nn.Dropout()
            ))

        self.sequentials_1 = nn.ModuleList(self.sequentials_1)
        self.sequentials_2 = nn.ModuleList(self.sequentials_2)
        self.linear = nn.Sequential(
            nn.Linear(interm_dim * len(num_channels), 1)
        )

        self.apply(init_weights_apply)


    def forward(self, features):
        outs = []
        for i in range(len(features)):
            feature = features[i].view(features[i].size(0), -1)
            out = self.sequentials_1[i](feature)
            out = out.view(out.size(0), -1)
            out = self.sequentials_2[i](out)
            outs.append(out)

        out = self.linear(torch.cat(outs, 1))
        return out
    
    
        
class Custom_Module(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(Custom_Module, self).__init__()
        
        # same parameters of loss net
        feature_sizes = params['feature_sizes']
        num_channels = params['num_channels']
        interm_dim = params['interm_dim']

        self.module_list, self.sequentials_1, self.sequentials_2 = [], [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            out_features = n_c // (e_d // 2)
            self.sequentials_1.append(nn.Sequential(
                nn.Conv2d(n_c ,out_features, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(),
            ))
            self.sequentials_2.append(nn.Sequential(
                nn.Linear(n_c * (e_d // 2), interm_dim),
                nn.ReLU(),
                nn.BatchNorm1d(interm_dim)
            ))

        self.sequentials_1 = nn.ModuleList(self.sequentials_1)
        self.sequentials_2 = nn.ModuleList(self.sequentials_2)
        self.linear = nn.Sequential(
            nn.Linear(interm_dim * len(num_channels), 1)
        )
        
        self.apply(init_weights_apply)


    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.sequentials_1[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.sequentials_2[i](out)
            outs.append(out)

        out = self.linear(torch.cat(outs, 1))
        return out


class Custom_Module_2(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(Custom_Module_2, self).__init__()

        # same parameters of loss net
        feature_sizes = params['feature_sizes']
        num_channels = params['num_channels']
        interm_dim = params['interm_dim']

        self.module_list, self.sequentials_1, self.sequentials_2 = [], [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.sequentials_1.append(nn.Sequential(
                nn.Conv2d(n_c, n_c // (e_d // 2), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(n_c // (e_d // 2)),
                nn.ReLU(),
            ))
            self.sequentials_2.append(nn.Sequential(
                nn.Linear(n_c * (e_d // 2), interm_dim),
                nn.ReLU(),
            )),

        self.sequentials_1 = nn.ModuleList(self.sequentials_1)
        self.sequentials_2 = nn.ModuleList(self.sequentials_2)

        self.linear_embedds = nn.Sequential(nn.Linear(num_channels[-1], interm_dim), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(interm_dim * (len(num_channels) + 1), 1), nn.ReLU())

        self.apply(init_weights_apply)


    def forward(self, features, embedds):
        outs = []
        for i in range(len(features)):
            out = self.sequentials_1[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.sequentials_2[i](out)
            outs.append(out)
        out_embedds = self.linear_embedds(embedds)
        out = self.linear(torch.cat(outs + [out_embedds], 1))
        return out
    
    
    

class Custom_Module_3(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(Custom_Module_3, self).__init__()

        # same parameters of loss net
        feature_sizes = params['feature_sizes']
        num_channels = params['num_channels']
        interm_dim = params['interm_dim']

        self.module_list = []
        self.sequentials_1 = []
        self.sequentials_2 = []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.sequentials_1.append(nn.Sequential(
                nn.Conv2d(n_c, n_c // (e_d // 2), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(n_c // (e_d // 2)),
                nn.ReLU(),
            ))
            self.sequentials_2.append(nn.Sequential(
                nn.Linear(n_c * (e_d // 2), interm_dim),
                nn.ReLU(),
            )),

        self.sequentials_1 = nn.ModuleList(self.sequentials_1)
        self.sequentials_2 = nn.ModuleList(self.sequentials_2)

        self.linear_embedds = nn.Sequential(
            nn.Linear(num_channels[-1], interm_dim), nn.ReLU()
        )
        self.linear_concat = nn.Sequential(
            nn.Linear(interm_dim * len(num_channels), interm_dim), nn.ReLU()
        )
        self.classifier = nn.Linear(interm_dim * 2, 1)

        self.apply(init_weights_apply)


    def forward(self, features, embedds):
        outs = []
        for i in range(len(features)):
            out = self.sequentials_1[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.sequentials_2[i](out)
            outs.append(out)
        out_concat = self.linear_concat(torch.cat(outs, 1))
        out_embedds = self.linear_embedds(embedds)
        out = self.classifier(torch.cat([out_concat, out_embedds], 1))
        return out
    
    
class Custom_MLP_2(nn.Module):
    def __init__(self, in_dim):
        super(Custom_MLP_2, self).__init__()

        self.sequential1 = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(in_dim//2),
        )
        
        self.sequential2 = nn.Sequential(
            nn.Linear(in_dim//2, in_dim//4),
            nn.ReLU(),
            nn.BatchNorm1d(in_dim//4),
        )
        
        self.sequential3 = nn.Sequential(
            nn.Linear(in_dim//4, in_dim//8),
            nn.ReLU(),
            nn.BatchNorm1d(in_dim//8),
        )
        
        self.classifier = nn.Linear(in_dim//8, 1)
                
        self.apply(init_weights_apply)

    def forward(self, x): 
        out = self.sequential1(x)
        out = self.sequential2(out)
        out = self.sequential3(out)
        out = self.classifier(out)
        return out
    
  
class Custom_Huber_Loss(nn.Module):
    def __init__(self):
        super(Custom_Huber_Loss, self).__init__()
        
    def forward(self, input, target):
        loss = input - target
        abs_loss = torch.abs(loss.clone())
        delta = torch.mean(abs_loss.clone()).item()
        loss[loss < delta] = torch.pow(loss[loss < delta], 2)
        loss[loss >= delta] = delta * (torch.abs(loss[loss >= delta]) - 0.5 * delta)
        
        return loss



class GTG_Module(nn.Module):
    def __init__(self, params):
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
        
        #self.c_mod = Custom_Module(ll_p).to(self.device)
        #self.c_mod = Custom_Module_2(ll_p).to(self.device)
        #self.c_mod = Custom_Module_3(ll_p).to(self.device)
        
        #self.c_mod = Custom_MLP(ll_p).to(self.device)
        self.c_mod = Custom_MLP_2(ll_p['num_channels'][-1]).to(self.device)
        
        # reduction='none' for the strategy computation
        self.mse_loss = nn.MSELoss(reduction='none').to(self.device)
        self.l1loss = nn.L1Loss(reduction='none').to(self.device)
        self.huber_loss = nn.HuberLoss(reduction='none').to(self.device)
        self.c_huber_loss = Custom_Huber_Loss().to(self.device)
    
    
    
    def define_additional_parameters(self, remaining_param: Dict[str, Any]) -> None:
        self.rbf_aff: bool = remaining_param['rbf_aff']
        self.A_function: str = remaining_param['A_function']
        self.ent_strategy: Entropy_Strategy = remaining_param['ent_strategy']
        self.threshold_strategy: str = remaining_param['threshold_strategy']
        self.threshold: float = remaining_param['threshold']
        
    
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
        A = torch.exp(A_m_2 / sigma_mm) 
        return A.to(self.device)
    
        
    def get_A_e_d(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A = torch.cdist(concat_embedds, concat_embedds).to(self.device)
        return A
    

    def get_A_cos_sim(self, concat_embedds: torch.Tensor) -> torch.Tensor:        
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(self.device)
        A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device)
        A_flat = A.flatten()
        A_flat[::(A.shape[-1]+1)] = 0
        return F.relu(A.clone())
        
        
    def get_A_corr(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A = F.relu(torch.corrcoef(concat_embedds).to(self.device))
        return A
    
    
    def get_A(self, embedding: torch.Tensor) -> None:

        # compute the affinity matrix
        if self.A_function != None:
            A = self.get_A_rbfk(embedding) if self.rbf_aff else self.get_A_fn[self.A_function](embedding)
        else: raise AttributeError('A Fucntion is None')
        
        assert torch.all(A >= 0), 'Negative value in self.A'

        if self.threshold_strategy != None and self.threshold != None:
            if self.A_function != 'e_d': A = torch.where(A < self.get_A_treshold(A), 0, A)
            else: A = torch.where(A > self.get_A_treshold(A), 1, A)
            
        assert torch.all(A >= 0), 'Negative value in self.A'
        assert torch.all(A <= 1), 'Values greater than 1'

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
        
        

    def get_X(self) -> None:
        self.X: torch.Tensor = torch.zeros((self.batch_size, self.n_classes), dtype=torch.float32, device=self.device)
        for idx, label in zip(self.labeled_indices, self.lab_labels):
            self.X[idx][int(label.item())] = 1.
        for idx in self.unlabeled_indices:
            for label in range(self.n_classes): self.X[idx][label] = 1. / self.n_classes
        #self.X.requires_grad_(True)

                 
    @torch.no_grad()
    def check_increasing_sum(self, mult_X_A_X: torch.Tensor, old_rowsum_X: int) -> int:
        rowsum_X = torch.sum(mult_X_A_X).item()
        if rowsum_X < old_rowsum_X: # it has to be increasing or at least equal
            logger.exception('Sum of the vector on the denominator is lower than the previous step')
            raise Exception('Sum of the vector on the denominator is lower than the previous step')
        return int(rowsum_X)

            
            
    def graph_trasduction_game(self, embedding) -> None:
        
        self.get_A(embedding)
        #self.A.register_hook(lambda t: print(f'hook self.A :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
        self.get_X()
        
        assert torch.all(self.A >= 0), 'Negative value in self.A'
        assert torch.all(self.A <= 1), 'Values greater than 1'
        
        assert torch.all(self.X >= 0), 'Negative value in self.X'
        
        err = float('Inf')
        i = 0
        old_rowsum_X = 0
        
        #unlab_entropy_hist_while = torch.clone(self.unlab_entropy_hist)
        
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = torch.clone(self.X)
            assert torch.all(X_old >= 0), 'Negative values in X_old'
            
            mm_A_X = torch.matmul(self.A, self.X)
            #mm_A_X.register_hook(lambda t: print(f'hook mm_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            assert torch.all(mm_A_X >= 0), 'Negative values in mm_A_X'
            
            mult_X_A_X = self.X * mm_A_X
            #mult_X_A_X.register_hook(lambda t: print(f'hook mult_X_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            assert torch.all(mult_X_A_X >= 0), 'Negative values in mult_X_A_X'
                        
            old_rowsum_X = self.check_increasing_sum(mult_X_A_X, old_rowsum_X)
            
            sum_X_A_X = torch.sum(mult_X_A_X, dim=1, keepdim=True)
            #sum_X_A_X.register_hook(lambda t: print(f'hook sum_X_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            assert torch.all(sum_X_A_X > 0), 'Negative or zero values in sum_X_A_X'
            
            div_sum_X_A_X = mult_X_A_X / sum_X_A_X
            #div_sum_X_A_X.register_hook(lambda t: print(f'hook div_sum_X_A_X :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
            assert torch.all(div_sum_X_A_X >= 0), 'Nagative values in div_sum_X_A_X'    
            
            
            iter_entropy = entropy(div_sum_X_A_X).to(self.device)
            assert torch.all(iter_entropy >= 0), 'Nagative values in iter_entropy'
            
            #unlab_entropy_hist_while[:, i] = iter_entropy
            self.unlab_entropy_hist[:, i] = iter_entropy
            
            i += 1
            err = torch.norm(div_sum_X_A_X.detach() - X_old)            
            self.X = div_sum_X_A_X
            
        #self.unlab_entropy_hist = unlab_entropy_hist_while
        #self.unlab_entropy_hist.register_hook(lambda t: print(f'hook self.unlab_entropy_hist :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
        assert torch.all(self.unlab_entropy_hist >= 0), 'Nagative values in self.unlab_entropy_hist'
        
        
        
        
        
        #self.unlab_entropy_hist.requires_grad_(True)
        
        
        
        
        
    # List[torch.Tensor] -> detection
    def preprocess_inputs(self, embedding: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        self.batch_size = len(embedding)
        self.n_lab_obs = int(self.batch_size * self.perc_labeled_batch) 
                
        shuffled_indices = torch.randperm(self.batch_size)

        self.labeled_indices: List[int] = shuffled_indices[:self.n_lab_obs].tolist()
        self.unlabeled_indices: List[int] = shuffled_indices[self.n_lab_obs:].tolist()
                        
        self.unlab_entropy_hist = torch.zeros((self.batch_size, self.gtg_max_iter), device=self.device)#, requires_grad=True)        
        
        mask = torch.zeros(self.batch_size, device=self.device)
        mask[self.labeled_indices] = 1.
        #mask.requires_grad_(True)

        
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

        #y_pred = self.c_mod(features, embedds).squeeze()
        #y_pred = self.c_mod(features).squeeze()
        y_pred = self.c_mod(embedds).squeeze()
        #y_pred.register_hook(lambda t: print(f'hook y_pred :\n {t} - {torch.any(torch.isnan(t))} - {torch.isfinite(t).all()} - {t.sum()}'))
        
        y_true, mask = self.preprocess_inputs(embedds, labels)
        #logger.info(f'y_pred {y_pred}\ny_true {y_true}')
        
        #loss_lab = self.mse_loss(y_pred[self.labeled_indices], y_true[self.labeled_indices])
        #loss_unlab = self.mse_loss(y_pred[self.unlabeled_indices], y_true[self.unlabeled_indices])
        #logger.info(f'loss_lab => {loss_lab}\tloss_unlab => {loss_unlab}')
        #loss = loss_lab + loss_unlab
        
        #loss = self.huber_loss(y_pred, y_true)
        loss = self.mse_loss(y_pred, y_true)
        #loss = self.c_huber_loss(y_pred, y_true)
        #return (loss_lab, loss_unlab), mask.bool()
        return loss, mask.bool()
