
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import entropy, init_weights_apply

from typing import Any, List, Tuple, Dict

import logging
logger = logging.getLogger(__name__)



class Module_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional, device):
        super(Module_LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        
        in_features_cls = hidden_size * (2 if bidirectional else 1) #num_layers
        self.hiddens_dim = num_layers * (2 if bidirectional else 1)
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        
        '''self.classifier = nn.Sequential(
            nn.Linear(in_features_cls, in_features_cls // 2), nn.ReLU(), 
            nn.Linear(in_features_cls // 2, in_features_cls // 4), nn.ReLU(), 
            nn.Linear(in_features_cls // 4, output_size)
        )'''
        self.classifier = nn.Linear(in_features_cls, output_size)
        
    def forward(self, x):
        hiddens = (
            torch.zeros(self.hiddens_dim, x.shape[0], self.hidden_size, device=self.device),
            torch.zeros(self.hiddens_dim, x.shape[0], self.hidden_size, device=self.device)
        )
        out, _ = self.lstm(x, hiddens)
        out = self.classifier(out)
        return out#.mean(dim=1)
    
    
'''class Multi_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, output_size=1):
        super(Multi_LSTM, self).__init__()
        
        self.device = device
        self.hidden_size = hidden_size
        self.lstms = nn.ModuleList([nn.LSTM(input_size=input_size, bidirectional=True, hidden_size=hidden_size, num_layers=num_layers, batch_first=True).to(self.device) for _ in range(4)])
        self.classifier = nn.Sequential( nn.Linear(4 * hidden_size, output_size) )
        
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        outs = []
        for i in range(len(features)):
            hiddens = (
                torch.zeros(2, self.hidden_size, device=self.device), 
                torch.zeros(2, self.hidden_size, device=self.device)
            )
            out, _ = self.lstms[i](features[i], hiddens)
            logger.info(out.shape)
            outs.append(out)
        
        out = self.classifier(torch.cat(outs, 1))
        return out'''
    

class Module_MLP_LSTM(nn.Module):
    def __init__(self, params: Dict[str, Any], device):
        super(Module_MLP_LSTM, self).__init__()
        
        self.device = device
        
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]
        
        self.linears, self.gaps, self.lstms = [], [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.gaps.append(nn.AvgPool2d(e_d))
            self.linears.append(nn.Sequential(
                nn.Linear(n_c, interm_dim), nn.ReLU(),# nn.BatchNorm1d(interm_dim)
            ))
            self.lstms.append(nn.LSTM(input_size=interm_dim, hidden_size=interm_dim, bidirectional=True, num_layers=1, batch_first=True))

        self.linears = nn.ModuleList(self.linears)
        self.gaps = nn.ModuleList(self.gaps)
        self.lstms = nn.ModuleList(self.lstms)
        dim_lin = len(num_channels) * interm_dim
        self.classifier = nn.Sequential(
            nn.Linear(dim_lin * 2, dim_lin), nn.ReLU(),
            nn.Linear(dim_lin, dim_lin//2), nn.ReLU(),
            nn.Linear(dim_lin//2, dim_lin//4), nn.ReLU(),
            nn.Linear(dim_lin//4, 1),
        )
        
        self.hiddens = (
            torch.zeros(2, interm_dim, device=self.device), 
            torch.zeros(2, interm_dim, device=self.device)
        )

    def forward(self, features):
        outs = []
        for i in range(len(features)):
            hiddens = (self.hiddens[0].clone(), self.hiddens[1].clone())
            out = self.gaps[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.linears[i](out)
            out = self.lstms[i](out, hiddens)[0]
            outs.append(out)
        out = self.classifier(torch.cat(outs, 1))
        return out




class Module_MLP(nn.Module):
    def __init__(self, gtg_iter, n_classes):
        super(Module_MLP, self).__init__()
        
        in_feat = gtg_iter * n_classes # -> [128 , n_classes * gtg_iter] = [128, 100] 
        
        # shared weights
        self.seq_linears = nn.Sequential(
            nn.Linear(in_feat, in_feat//2), nn.ReLU(),# nn.BatchNorm1d(in_feat//2),# nn.Dropout(), #nn.BatchNorm1d(in_feat//2),
            #nn.Linear(in_feat//2, in_feat//2), nn.ReLU(), nn.BatchNorm1d(in_feat//2),
            nn.Linear(in_feat//2, in_feat//4), nn.ReLU(),# nn.BatchNorm1d(in_feat//4),# nn.Dropout(), #nn.BatchNorm1d(in_feat//4),
            #nn.Linear(in_feat//4, in_feat//4), nn.ReLU(), nn.BatchNorm1d(in_feat//4),
            #nn.Linear(in_feat//4, in_feat//8), nn.ReLU(), nn.Dropout(), #nn.BatchNorm1d(in_feat//8),
            #nn.Linear(in_feat//8, 1), nn.ReLU(), #nn.BatchNorm1d(in_feat//4),# to test after
        )
        
        #self.linear = nn.Linear(in_feat//2 * 4, 1)
        self.linear = nn.Linear(in_feat//4 * 4, 1)
        #self.linear = nn.Linear(in_feat//8 * 4, 1)
        #self.linear = nn.Linear(4, 1)


    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = features[i].view(features[i].size(0), -1)
            out = self.seq_linears(out)
            outs.append(out)
        
        out = self.linear(torch.cat(outs, 1))
        return out

'''class MLP(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(MLP, self).__init__()

        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

        self.seq_linears = []
        self.convs = []
        
        for n_c, e_d in zip(num_channels, feature_sizes):
            size = n_c * e_d * e_d
            self.seq_linears.append(nn.Sequential(
                #65536, 32768, 16384, 8192 
                nn.Linear(size, size//10), nn.BatchNorm1d(size//10), nn.ReLU(),
                nn.Linear(size//10, interm_dim), nn.ReLU(), nn.BatchNorm1d(interm_dim), nn.ReLU()
            ))
            self.convs.append(nn.Conv2d(n_c, n_c, kernel_size=3, stride=1, padding=1))

        self.linears_1 = nn.ModuleList(self.seq_linears)
        self.convs = nn.ModuleList(self.convs)
        size = interm_dim * len(self.seq_linears)
        self.linears_2 = nn.Sequential( #262144
            nn.Linear(size, size//2), nn.ReLU(), nn.BatchNorm1d(size//2), nn.ReLU(),
            nn.Linear(size//2, size//4), nn.ReLU(), nn.BatchNorm1d(size//4), nn.ReLU(),
            nn.Linear(size//4, 1),
        )
        #self.linear = nn.Linear(interm_dim * len(self.seq_linears), 1)
        
        
    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.convs[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.linears_1[i](out)
            outs.append(out)
        
        out = self.linears_2(torch.cat(outs, 1))
        #out = self.linear(torch.cat(outs, 1))
        return out'''

'''def init_weights(net):
    for m in net.modules():
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)'''


'''class MLP(nn.Module):
    def __init__(self, embedd_dim):
        super(MLP, self).__init__()

        self.linears = nn.Sequential(
            nn.Linear(embedd_dim, embedd_dim//2), nn.ReLU(), # nn.BatchNorm1d(embedd_dim//2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(embedd_dim//2, embedd_dim//4), nn.ReLU(),# nn.BatchNorm1d(embedd_dim//4), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(embedd_dim//4, embedd_dim//8), nn.ReLU(),# nn.BatchNorm1d(embedd_dim//8), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(embedd_dim//8, embedd_dim//16), nn.ReLU(),# nn.BatchNorm1d(embedd_dim//16), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(embedd_dim//16, 1),
        )
        
        #self.apply(init_weights)
    
    def forward(self, embedds): # [bs, 512, 4, 4]
        embedds = embedds.view(embedds.size(0), -1) # [bs, 512*4*4] -> [bs, 8192]
        return self.linears(embedds)'''
        

class CNN_MLP(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(CNN_MLP, self).__init__()
        
        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

        self.convs, self.linears = [], []
        dim_class = interm_dim * len(num_channels)

        for n_c, e_d in zip(num_channels, feature_sizes):
            out_features = n_c // (e_d // 2)
            self.convs.append(nn.Sequential(
                nn.Conv2d(n_c ,out_features, kernel_size=3, stride=2, padding=1), # instead og GAP of LL_Module
                nn.BatchNorm2d(out_features),
                nn.ReLU(),
            ))
            self.linears.append(nn.Sequential(
                nn.Linear(n_c * (e_d // 2), interm_dim), # same dimension of the LL_Module latent space
                nn.ReLU(),
            ))

        self.convs = nn.ModuleList(self.convs)
        self.linears = nn.ModuleList(self.linears)
        self.regressor = nn.Sequential(
            nn.Linear(dim_class, dim_class//2), nn.ReLU(),
            nn.Linear(dim_class//2, dim_class//4 ), nn.ReLU(),
            nn.Linear(dim_class//4, 1)
        )
        

    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.convs[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.linears[i](out)
            outs.append(out)

        out = self.regressor(torch.cat(outs, 1))
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
                nn.Linear(n_c, interm_dim), nn.ReLU(), #nn.BatchNorm1d(interm_dim)
            ))

        self.linears = nn.ModuleList(self.linears)
        self.gaps = nn.ModuleList(self.gaps)


    def forward(self, features, id_feat):
        out = self.gaps[id_feat](features)
        out = out.view(out.size(0), -1)
        return self.linears[id_feat](out)
    
    
class LL_Module(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(LL_Module, self).__init__()

        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

        self.linears, self.gaps = [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.gaps.append(nn.AvgPool2d(e_d))
            self.linears.append(nn.Sequential(
                nn.Linear(n_c, interm_dim), nn.ReLU(),
            ))

        self.linears = nn.ModuleList(self.linears)
        self.gaps = nn.ModuleList(self.gaps)
        self.classifier = nn.Linear(interm_dim * len(num_channels), 1)


    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.gaps[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.linears[i](out)
            outs.append(out)
        out = torch.cat(outs, 1)
        return self.classifier(out)


        

class GTGModule(nn.Module):
    def __init__(self, params):
        super(GTGModule, self).__init__()
        
        self.name = self.__class__.__name__

        gtg_p, ll_p = params

        self.get_A_fn = { 'cos_sim': self.get_A_cos_sim, 'corr': self.get_A_corr, 'rbfk': self.get_A_rbfk, }
        
        self.gtg_tol: float = gtg_p["gtg_t"]
        self.gtg_max_iter: int = gtg_p["gtg_i"]
        
        self.AM_strategy: str = gtg_p["am_s"]
        
        self.AM_threshold: float = gtg_p["am_t"]
        
        self.list_AM_threshold_strategy: List[str] = gtg_p["am_ts"]
        self.list_AM_function: List[str] = gtg_p["am"]
        
        self.ent_strategy: str = gtg_p["e_s"]
        self.perc_labelled_batch: int = gtg_p["plb"]
        
        self.n_top_k_obs: int = gtg_p["n_top_k_obs"]
        self.n_classes: int = gtg_p["n_classes"]
        self.device: int = gtg_p["device"]

        #self.mod_lstm = Module_LSTM(input_size=self.gtg_max_iter, hidden_size=self.gtg_max_iter, num_layers=1, output_size=1).to(self.device)
        self.cnn_mlp = CNN_MLP(ll_p).to(self.device)
        #self.ll_mod = LL_Module(ll_p).to(self.device)
        '''self.single_lstm = Module_LSTM(
            input_size=self.gtg_max_iter, #ll_p["num_channels"][-1], 
            hidden_size=self.gtg_max_iter, #ll_p["num_channels"][-1], 
            num_layers=1, 
            output_size=1, #self.gtg_max_iter,
            bidirectional=True,
            device=self.device
        ).to(self.device)'''
        
    def define_idx_params(self, id_am_ts: int, id_am: int) -> None: # in order to define the indices of AM_threshold_strategy and AM_function
        self.AM_threshold_strategy: str = self.list_AM_threshold_strategy[id_am_ts]
        self.AM_function: str = self.list_AM_function[id_am]

    
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
        return torch.clamp(A, min=0., max=1.)
    

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
    
    
    def get_A(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.AM_function != None: A = self.get_A_fn[self.AM_function](embedding) # compute the affinity matrix
        else: raise AttributeError('A Fucntion is None')
                
        if self.AM_threshold_strategy != None and self.AM_threshold!= None:
            if self.AM_function != 'rbfk': A = torch.where(A < self.get_A_treshold(A), 0, A)
            else: A = torch.where(A > self.get_A_treshold(A), 1, A)

        if self.AM_strategy == 'diversity':
            A = 1 - A # set the whole matrix as a distance matrix and not similarity matrix
        elif self.AM_strategy == 'mixed':    
            if self.AM_function == 'rbfk': A = 1 - A # set the unlabelled submatrix as distance matrix and not similarity matrix
            A[:self.n_lab_obs, :self.n_lab_obs] = 1 - A[:self.n_lab_obs, :self.n_lab_obs] # -> LL distance
        
        return A
        

    #def get_X(self, probs: torch.Tensor) -> None:
    def get_X(self) -> None:
        self.X: torch.Tensor = torch.zeros((self.batch_size, self.n_classes), dtype=torch.float32, device=self.device)
       
        self.X[torch.arange(self.n_lab_obs), self.lab_labels] = 1.
        #self.X[torch.arange(self.n_lab_obs), :] = probs[torch.arange(self.n_lab_obs)]

        self.X[self.n_lab_obs:, :] = torch.ones(self.n_classes, device=self.device) * (1. / self.n_classes)
        #self.X.requires_grad_(True)

        
    @torch.no_grad()
    def graph_trasduction_game_detached(self, embedding: torch.Tensor) -> torch.Tensor:
        
        entropy_hist = torch.zeros((self.batch_size, self.gtg_max_iter), device=self.device)        
        A = self.get_A(embedding).detach()
                
        err = float('Inf')
        i = 0
        X = self.X.clone()
        
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = X.clone()
            
            X *= torch.mm(A, X)
            X /= torch.sum(X, dim=1, keepdim=True)
            
            entropy_hist[self.n_lab_obs:, i] = entropy(X)[self.n_lab_obs:].to(self.device)
            #entropy_hist[:, i] = entropy(X).to(self.device)
            
            err = torch.norm(X - X_old)
            i += 1
        return entropy_hist

    
    
    def graph_trasduction_game_simplier(self, features_embedding: torch.Tensor) -> torch.Tensor:
                        
        entropy_hist = torch.zeros((self.batch_size, self.gtg_max_iter), device=self.device)        
        
        A = self.get_A(features_embedding) # -> nxn
        X = self.X.clone() # -> nxm
        if A.requires_grad: X.requires_grad_(True)
        Xs = torch.empty((self.batch_size, self.n_classes, 0), device=self.device, dtype=torch.float32, requires_grad=True if A.requires_grad else False)        
        
        err = float('Inf')
        i = 0
                        
        while err > self.gtg_tol and i < self.gtg_max_iter:
            
            if X.requires_grad: X.register_hook(lambda grad: logger.info(f'{i} -- 1- X grad: {grad.sum()}'))
            
            X_old = X.detach().clone()
                        
            X = X * torch.mm(A, X)
            if X.requires_grad: X.register_hook(lambda grad: logger.info(f'{i} -- 2- X grad: {grad.sum()}'))
            X = X / torch.sum(X, dim=1, keepdim=True)
            if X.requires_grad: X.register_hook(lambda grad: logger.info(f'{i} -- 3- X grad: {grad.sum()}'))
            
            entropy_hist[self.n_lab_obs:, i] = entropy(X.detach())[self.n_lab_obs:].to(self.device)

            err = torch.norm(X.detach() - X_old)
            Xs = torch.cat((Xs, X.unsqueeze(dim=-1)), dim=-1)
            
            i += 1
        
        # fill in case we early extit by the norm err
        for _ in range(self.gtg_max_iter - Xs.shape[-1]): 
            Xs = torch.cat((
                Xs, torch.zeros((self.batch_size, self.n_classes, 1), device=self.device, dtype=torch.float32, requires_grad=True if A.requires_grad else False)
            ), dim=-1)

        return Xs, entropy_hist
    
    

    def graph_trasduction_game(self, features_embedding: torch.Tensor) -> torch.Tensor:
                        
        entropy_hist = torch.zeros((self.batch_size, self.gtg_max_iter), device=self.device)        
        
        A = self.get_A(features_embedding) # -> nxn
        X = self.X.clone() # -> nxm
        if A.requires_grad: X.requires_grad_(True)
        Xs = torch.empty((self.batch_size, self.n_classes, 0), device=self.device, dtype=torch.float32, requires_grad=True if A.requires_grad else False)        
        
        err = float('Inf')
        i = 0
                        
        while err > self.gtg_tol and i < self.gtg_max_iter:
            
            if X.requires_grad: X.register_hook(lambda grad: logger.info(f'X grad: {grad.sum()}'))
            
            X_old = X.detach().clone()
                        
            mm_A_X = torch.mm(A, X) # -> nxm
            if mm_A_X.requires_grad: mm_A_X.register_hook(lambda grad: logger.info(f'mm_A_X grad: {grad.sum()}'))
            
            mult_X_A_X = X * mm_A_X
            if mult_X_A_X.requires_grad: mult_X_A_X.register_hook(lambda grad: logger.info(f'mult_X_A_X grad: {grad.sum()}'))
            
            sum_X_A_X = torch.sum(mult_X_A_X, dim=1, keepdim=True)
            if sum_X_A_X.requires_grad: sum_X_A_X.register_hook(lambda grad: logger.info(f'sum_X_A_X grad: {grad.sum()}'))
            
            div_sum_X_A_X = mult_X_A_X / (sum_X_A_X + 1e-8)  # Add a small epsilon to avoid division by zero, 1e-8
            if div_sum_X_A_X.requires_grad: div_sum_X_A_X.register_hook(lambda grad: logger.info(f'div_sum_X_A_X grad: {grad.sum()}'))
            
            #entropy_hist[self.n_lab_obs:, i] = entropy(div_sum_X_A_X.detach())[self.n_lab_obs:].to(self.device)
            entropy_hist[:, i] = entropy(div_sum_X_A_X.detach()).to(self.device)


            err = torch.norm(div_sum_X_A_X.detach() - X_old)
            Xs = torch.cat((Xs, div_sum_X_A_X.unsqueeze(dim=-1)), dim=-1)

            X = X + div_sum_X_A_X
            #X = X + div_sum_X_A_X
            #X = X / torch.sum(X, dim=1, keepdim=True)
            
            i += 1
        
        # fill in case we early extit by the norm err
        for _ in range(self.gtg_max_iter - Xs.shape[-1]): 
            Xs = torch.cat((
                Xs, torch.zeros((self.batch_size, self.n_classes, 1), device=self.device, dtype=torch.float32, requires_grad=True if A.requires_grad else False)
            ), dim=-1)

        return Xs, entropy_hist
    
          
        
    # List[torch.Tensor] -> detection
    def get_entropies(self, embedding: torch.Tensor) -> torch.Tensor: #Tuple[torch.Tensor, torch.Tensor]:     
                  
        entropy_hist = self.graph_trasduction_game_detached(embedding)
        #entropy_hist = self.graph_trasduction_game(embedding)[1]
 
        if self.ent_strategy == 'mean': quantity_result = torch.mean(entropy_hist, dim=1)
        # computing the mean of the entropis history    
        elif self.ent_strategy == 'integral': quantity_result = torch.trapezoid(entropy_hist, dim=1)
        # computing the area of the entropis history using trapezoid formula    
        else:
            logger.exception(' Invalid GTG Strategy') 
            raise AttributeError(' Invalid GTG Strategy')
            
        return quantity_result
    
    
    #labelled_dim = -1
    def forward(self, features: List[torch.Tensor], embedds: torch.Tensor, outs: torch.Tensor, labels: torch.Tensor, weight: int = 1) \
        -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: 
        
        self.batch_size = len(embedds)
        self.n_lab_obs = int(self.batch_size * self.perc_labelled_batch)
        self.lab_labels = labels[:self.n_lab_obs]
        
        self.get_X()
        #self.get_X(F.softmax(outs, dim=1))
        
        '''latent_features = [ ]
        for id, features_embedding in enumerate(features):
            emb_ls = self.mod_ls(features_embedding, id)
            if weight == 0: emb_ls = emb_ls.detach()
            latent_features.append(self.graph_trasduction_game(emb_ls)[0]) # -> takes always the first Xs
            
        y_pred = self.mod_mlp(latent_features).squeeze()'''
        
        #LSTM MODULE
        '''latent_features = [ ]
        for id, features_embedding in enumerate(features):
            emb_ls = self.mod_ls(features_embedding, id)
            if weight == 0: emb_ls = emb_ls.detach()
            latent_features.append(emb_ls)
        concat_ls = torch.cat(latent_features, dim=1)'''
        
        #y_pred = self.ll_mod(features).squeeze()
        y_pred = self.cnn_mlp(features).squeeze()
        
        #LSTM MODULE
        '''pred_Xs = self.graph_trasduction_game(concat_ls)[0]
        pred_Xs = entropy(pred_Xs, dim=1).unsqueeze(dim=1)
        
        #y_pred = self.mod_mlp(latent_features).squeeze()
        y_pred = self.single_lstm(pred_Xs).squeeze() 
        '''

        if weight == 0: y_pred = y_pred.detach()

        if torch.any(torch.isnan(y_pred)):
            print(features)
            print(y_pred)
        assert not torch.any(torch.isnan(y_pred)), 'y_pred is nan'
        
        y_true = self.get_entropies(embedds.detach())
        
        if torch.any(torch.isnan(y_true)):
            print(embedds)
            print(y_true)
        assert not torch.any(torch.isnan(y_true)), 'y_true is nan'

        if self.training:
            labelled_mask = torch.zeros(self.batch_size, device=self.device)
            labelled_mask[:self.n_lab_obs] = 1.
            return (y_pred, y_true), labelled_mask.bool()
        
        else: return (y_pred, y_true), None

        #LSTM MODULE
'''self.batch_size = len(embedds)
        self.n_lab_obs = int(self.batch_size * self.perc_labelled_batch) if labelled_dim == -1 else labelled_dim
        self.lab_labels = labels[:self.n_lab_obs]
        
        self.get_X(F.softmax(outs, dim=1))
        
        y_pred = torch.clamp(self.mod_lstm(self.graph_trasduction_game(embedds)[1]).squeeze(), 0., 1.)

        y_true = torch.zeros(self.batch_size, device=self.device)
        y_true[:self.n_lab_obs] = 1.
        
        if self.training: return (y_pred, y_true), y_true.bool()
        else: return (y_pred, y_true), None'''


        #y_pred = torch.mean(
        #    torch.cat([self.preprocess_inputs(self.mod_ls(feature, id))[1].unsqueeze(dim=0) for id, feature in enumerate(features)], dim=0)
        #, dim=0)
        
'''y_pred = self.mod_mlp(
            [            #[n x 128]          +               #[n x 1]           =           [n x 129]
                torch.cat((feature, self.preprocess_inputs(feature)[1].view(-1,1)), dim=1) # [n x 1] each
                for feature in [self.mod_ls(feature, id) for id, feature in enumerate(features)] # -> [n x 128]
            ]
        ).squeeze()'''
        
        #y_pred = self.mod_lstm(self.self.preprocess_inputs(embedds)[2])

        
        
        #y_pred = self.mlp(features).squeeze()
'''y_pred = self.mlp_e(embedds).squeeze()
            
        labels = labels.to(self.device)
        self.batch_size = len(embedds)
            
        indices = torch.arange(self.batch_size)
        self.n_lab_obs = int(self.batch_size * self.perc_labelled_batch)
            
        self.labelled_indices: torch.Tensor = indices[:self.n_lab_obs].to(self.device)
        self.unlabelled_indices: torch.Tensor = indices[self.n_lab_obs:].to(self.device)
            
        self.lab_labels = labels[self.labelled_indices]
            
        self.get_X(F.softmax(outs, dim=1))
            
        y_true = self.preprocess_inputs(embedds)[1]'''
            
