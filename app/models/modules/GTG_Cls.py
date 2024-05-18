
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import entropy

from typing import Any, List, Tuple, Dict

import logging
logger = logging.getLogger(__name__)



class Module_MLP(nn.Module):
    #def __init__(self, n_classes, gtg_iter):
    def __init__(self, interm_dim):
        super(Module_MLP, self).__init__()       
        
        #in_feat = n_classes * gtg_iter 
        in_feat = interm_dim + 1 # interm_dim + mean entropy 
        
        self.sqe_linears = [
            nn.Sequential(
                nn.Linear(in_feat, in_feat), nn.BatchNorm1d(in_feat), nn.ReLU(), 
                nn.Linear(in_feat, in_feat//2), nn.BatchNorm1d(in_feat//2), nn.ReLU(),
                nn.Linear(in_feat//2, in_feat//2), nn.BatchNorm1d(in_feat//2), nn.ReLU(),
                nn.Linear(in_feat//2, in_feat//4), nn.BatchNorm1d(in_feat//4), nn.ReLU(),
                nn.Linear(in_feat//4, in_feat//4), nn.BatchNorm1d(in_feat//4), nn.ReLU(),
            ) for _ in range(4)
        ]
        
        self.linears = nn.ModuleList(self.sqe_linears)
        self.linear = nn.Linear(in_feat//4 * len(self.sqe_linears), 1)


    def forward(self, features):
        outs = []
        for i in range(len(features)):
            #out = X_features[i].view(X_features[i].size(0), -1) # -> [n, ...]
            out = self.linears[i](features[i])
            outs.append(out)
        
        out = self.linear(torch.cat(outs, 1))
        return out



class Module_LS(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(Module_LS, self).__init__()

        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

        self.linears, self.gaps = [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.gaps.append(nn.AvgPool2d(e_d))
            self.linears.append(nn.Sequential(
                nn.Linear(n_c, interm_dim),
                nn.ReLU(),
            ))

        self.linears = nn.ModuleList(self.linears)
        self.gaps = nn.ModuleList(self.gaps)


    def forward(self, features, id_feat):
        out = self.gaps[id_feat](features)
        out = out.view(out.size(0), -1)
        return self.linears[id_feat](out)

    

class GTGModule(nn.Module):
    def __init__(self, params):
        super(GTGModule, self).__init__()
        
        self.name = self.__class__.__name__

        gtg_p, ll_p = params

        self.get_A_fn = { 'cos_sim': self.get_A_cos_sim, 'corr': self.get_A_corr, 'rbfk': self.get_A_rbfk, }
        
        self.gtg_tol: float = gtg_p["gtg_t"]
        self.gtg_max_iter: int = gtg_p["gtg_i"]
        
        self.AM_strategy: str = gtg_p["am_s"]
        self.AM_threshold_strategy: str = gtg_p["am_ts"]
        self.AM_threshold: float = gtg_p["am_t"]
        
        self.ent_strategy: str = gtg_p["e_s"]
        self.perc_labelled_batch: int = gtg_p["plb"]
        
        self.n_top_k_obs: int = gtg_p["n_top_k_obs"]
        self.n_classes: int = gtg_p["n_classes"]
        self.device: int = gtg_p["device"]

        self.mod_ls = Module_LS(ll_p).to(self.device)
        self.mod_mlp = Module_MLP(ll_p["interm_dim"]).to(self.device)
        #self.mod_mlp = Module_MLP(self.n_classes, self.gtg_max_iter).to(self.device)
        
        
        
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
        
        #A = torch.exp(-A_matrix.pow(2) / (torch.mm(sigmas.T, sigmas))).to(self.device)      
        #return torch.clamp(A, min=0., max=1.)
        
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
        
        #A.fill_diagonal_(1.)
        #return torch.clamp(A, min=0., max=1.)
        
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
            # set the unlabelled submatrix as distance matrix and not similarity matrix

            if self.AM_function != 'rbfk': A = 1 - A # -> all distance matrix
            A[self.labelled_indices[:, None], self.labelled_indices] = 1 - A[self.labelled_indices[:, None], self.labelled_indices]
        
        self.A = A
        
        

    def get_X(self, probs: torch.Tensor) -> None:
        self.X: torch.Tensor = torch.zeros((self.batch_size, self.n_classes), dtype=torch.float32, device=self.device)
       
        ##################################################################
        #self.X[self.labelled_indices, self.lab_labels] = 1.
        self.X[self.labelled_indices, :] = probs[self.labelled_indices]
        ##################################################################
        
        self.X[self.unlabelled_indices, :] = torch.ones(self.n_classes, device=self.device) * (1./self.n_classes)
        self.X.requires_grad_(True)



    @torch.no_grad()
    def graph_trasduction_game_detached(self, embedding: torch.Tensor) -> torch.Tensor:
        
        entropy_hist = torch.zeros((self.batch_size, self.gtg_max_iter), device=self.device)        
        self.get_A(embedding)
                
        err = float('Inf')
        i = 0
        X = self.X.clone()
                
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = torch.clone(X)
            
            X *= torch.mm(self.A, X)
            X /= torch.sum(X, dim=1, keepdim=True)
            
            entropy_hist[:, i] = entropy(X).to(self.device)
            
            err = torch.norm(X - X_old)
            i += 1
        return entropy_hist



    def graph_trasduction_game(self, embedding: torch.Tensor) -> torch.Tensor:
                        
        entropy_hist = torch.zeros((self.batch_size, self.gtg_max_iter), device=self.device,)# requires_grad=True)        
        
        self.get_A(embedding)
        X = self.X.clone()
        self.Xs = torch.empty((self.batch_size, self.n_classes, 0), device=self.device, requires_grad=True)        
        
        err = float('Inf')
        i = 0
                
        while err > self.gtg_tol and i < self.gtg_max_iter:
            
            X_old = X.detach().clone()
            
            mm_A_X = torch.mm(self.A, X)
            mult_X_A_X = X * mm_A_X
            sum_X_A_X = torch.sum(mult_X_A_X, dim=1, keepdim=True)# + 1e-8
            div_sum_X_A_X = mult_X_A_X / sum_X_A_X
            
            entropy_hist[:, i] = entropy(div_sum_X_A_X.detach()).to(self.device)
            
            err = torch.norm(div_sum_X_A_X.detach() - X_old)
            X = div_sum_X_A_X
            self.Xs = torch.cat((self.Xs, X.unsqueeze(dim=-1)), dim=-1)
                        
            i += 1
        
        # fill in case we early extit by the norm err
        for _ in range(self.gtg_max_iter - self.Xs.shape[-1]): 
            self.Xs = torch.cat((
                self.Xs, torch.zeros(self.batch_size, self.n_classes, 1, device=self.device, dtype=torch.float32, requires_grad=True)
            ), dim=-1)
                           
        return entropy_hist.requires_grad_(True)
    
        
        
    # List[torch.Tensor] -> detection
    def preprocess_inputs(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:     
                
        entropy_hist = self.graph_trasduction_game(embedding)

        if self.ent_strategy == 'mean': quantity_result = torch.mean(entropy_hist, dim=1)
        # computing the mean of the entropis history    
        elif self.ent_strategy == 'integral': quantity_result = torch.trapezoid(entropy_hist, dim=1)
        # computing the area of the entropis history using trapezoid formula    
        else:
            logger.exception(' Invlaid GTG Strategy') 
            raise AttributeError(' Invlaid GTG Strategy')

        return self.Xs, quantity_result
    
    
    
    def forward(self, features: List[torch.Tensor], embedds: torch.Tensor, outs: torch.Tensor, labels) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: 
                    
        self.batch_size = len(embedds)
        
        if self.training:
            indices = torch.randperm(self.batch_size) # -> unique for each batch
            self.n_lab_obs = int(self.batch_size * self.perc_labelled_batch)
        else: 
            logger.info('evaluation mode')
            indices = torch.arange(self.batch_size)
            self.n_lab_obs = self.batch_size // 2
        
        self.labelled_indices: torch.Tensor = indices[:self.n_lab_obs].to(self.device)
        self.unlabelled_indices: torch.Tensor = indices[self.n_lab_obs:].to(self.device)
        
        self.lab_labels = labels[self.labelled_indices]
        
        self.get_X(F.softmax(outs, dim=1))
        

        '''y_pred = self.mod_mlp(
            [self.preprocess_inputs(self.graph_trasduction_game, self.mod_gap(feature, id))[0] for id, feature in enumerate(features)]
        ).squeeze()'''
        
        '''y_pred = torch.mean(
            torch.cat([self.preprocess_inputs(self.graph_trasduction_game, self.mod_ls(feature, id))[1].unsqueeze(dim=0) for id, feature in enumerate(features)], dim=0)
        , dim=0)'''     
        
        
        y_pred = self.mod_mlp(
            [            #[n x 128]          +               #[n x 1]           =           [n x 129]
                torch.cat((feature, self.preprocess_inputs(self.graph_trasduction_game, feature)[1].view(-1,1)), dim=1) # [n x 1] each
                for feature in [self.mod_ls(feature, id) for id, feature in enumerate(features)] # -> [n x 128]
            ]
        ).squeeze()
            
            
        if self.training:
            labelled_mask = torch.zeros(self.batch_size, device=self.device)
            labelled_mask[self.labelled_indices] = 1.
            _, y_true = self.preprocess_inputs(self.graph_trasduction_game_detached, embedds)
            return (y_pred, y_true), labelled_mask.bool()
        
        else: return (y_pred, None), None
