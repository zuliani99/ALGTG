
import torch
import torch.nn as nn

from utils import init_weights_apply

from typing import Any, Dict

import logging
logger = logging.getLogger(__name__)



class Custom_MLP(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(Custom_MLP, self).__init__()

        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

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
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

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
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

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
            ))

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
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

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
            ))

        self.sequentials_1 = nn.ModuleList(self.sequentials_1)
        self.sequentials_2 = nn.ModuleList(self.sequentials_2)

        self.linear_embedds = nn.Sequential( nn.Linear(num_channels[-1], interm_dim), nn.ReLU() )
        self.linear_concat = nn.Sequential( nn.Linear(interm_dim * len(num_channels), interm_dim), nn.ReLU() )
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
            nn.BatchNorm1d(in_dim//2),
            nn.ReLU(),
        )
        
        self.sequential2 = nn.Sequential(
            nn.Linear(in_dim//2, in_dim//4),
            nn.BatchNorm1d(in_dim//4),
            nn.ReLU(),
        )
        
        self.sequential3 = nn.Sequential(
            nn.Linear(in_dim//4, in_dim//8),
            nn.BatchNorm1d(in_dim//8),
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(in_dim//8, 1)
                
        self.apply(init_weights_apply)

    def forward(self, x): 
        out = self.sequential1(x)
        out = self.sequential2(out)
        out = self.sequential3(out)
        out = self.classifier(out)
        return out
    
    
    
class Custom_Module(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(Custom_Module, self).__init__()

        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = num_channels[-1]

        self.module_list, self.sequentials_1, self.sequentials_2 = [], [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.sequentials_1.append(nn.Sequential(
                nn.Conv2d(n_c, n_c // (e_d // 2), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(n_c // (e_d // 2)),
                nn.ReLU(),
                
                nn.Conv2d(n_c // (e_d // 2), n_c // (e_d // 2), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(n_c // (e_d // 2)),
                nn.ReLU(),
            ))
            
            self.sequentials_2.append(nn.Sequential(
                nn.Linear(n_c * (e_d // 2), interm_dim),
                nn.ReLU(),
            ))

        self.sequentials_1 = nn.ModuleList(self.sequentials_1)
        self.sequentials_2 = nn.ModuleList(self.sequentials_2)

        self.linear_concat = nn.Sequential( nn.Linear(interm_dim * len(num_channels), interm_dim), nn.ReLU() )
        self.classifier = nn.Linear(interm_dim * 2, 1)



    def forward(self, features, embedds):
        outs = []
        for i in range(len(features)):
            out = self.sequentials_1[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.sequentials_2[i](out)
            outs.append(out)
        out_concat = self.linear_concat(torch.cat(outs, 1))
        out = self.classifier(torch.cat([out_concat, embedds], 1))
        return out

class Custom_conv_Module(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(Custom_conv_Module, self).__init__()

        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

        self.convs, self.linears = [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.convs.append(nn.Sequential(
                nn.Conv2d(n_c, n_c // (e_d // 2), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(n_c // (e_d // 2)),
                nn.ReLU(),
                
                nn.Conv2d(n_c // (e_d // 2), n_c // (e_d // 2), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(n_c // (e_d // 2)),
                nn.ReLU(),
            ))
            self.linears.append(nn.Sequential(
                nn.Linear(n_c * (e_d // 2), interm_dim),
                nn.ReLU(),
            ))

        self.convs = nn.ModuleList(self.convs)
        self.linears = nn.ModuleList(self.linears)

        self.linear = nn.Sequential(nn.Linear(interm_dim * len(num_channels), 1), nn.ReLU())


    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.convs[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.linears[i](out)
            outs.append(out)
        out = self.linear(torch.cat(outs, 1))
        return out
  
'''

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

'''


'''def get_A_rbfk(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        
        A_matrix = self.get_A_e_d(concat_embedds)
        seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1)[:, 6]]            

        sigmas = torch.unsqueeze(torch.tensor([
            self.get_A_fn[self.AM_function](torch.cat(( # type: ignore
                torch.unsqueeze(concat_embedds[i], dim=0), torch.unsqueeze(seventh_neigh[i], dim=0)
            )))[0,1].item()
            for i in range(concat_embedds.shape[0]) 
        ], device=self.device), dim=0)
        
        
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
        
        A_flat = A.flatten()
        A_flat[::(A.shape[-1]+1)] = 0.
        return torch.clamp(A.clone(), min=0., max=1.)'''

class Custom_GAP_Embedds_Module(nn.Module):
    def __init__(self, params):
        super(Custom_GAP_Embedds_Module, self).__init__()

        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

        self.convs, self.linears, self.gaps = [], [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.convs.append(nn.Sequential(
                nn.Conv2d(n_c, n_c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(n_c), nn.ReLU()
            ))
            self.gaps.append(nn.AvgPool2d(e_d))
            self.linears.append(nn.Sequential(nn.Linear(n_c, interm_dim), nn.ReLU()))

        self.convs = nn.ModuleList(self.convs)
        self.linears = nn.ModuleList(self.linears)
        self.gaps = nn.ModuleList(self.gaps)

        self.linear_embedds = nn.Sequential(nn.Linear(num_channels[-1], interm_dim), nn.ReLU())
        self.linear = nn.Linear(interm_dim * (len(num_channels) + 1), 1)


    def forward(self, features, embedds):
        outs = []
        for i in range(len(features)):
            out = self.convs[i](features[i])
            out = self.gaps[i](out)
            out = out.view(out.size(0), -1)
            out = self.linears[i](out)
            outs.append(out)
        outs.append(self.linear_embedds(embedds))
        out = self.linear(torch.cat(outs, 1))
        return out




'''class MLP(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(MLP, self).__init__()

        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

        self.seq_linears, self.gaps = [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.gaps.append(nn.AvgPool2d(e_d))
            self.seq_linears.append(nn.Sequential(
                nn.Linear(n_c, interm_dim), nn.BatchNorm1d(interm_dim), nn.ReLU(),
                nn.Linear(interm_dim, interm_dim//4), nn.BatchNorm1d(interm_dim//4), nn.ReLU(),
            ))

        self.linears = nn.ModuleList(self.seq_linears)
        self.gaps = nn.ModuleList(self.gaps)
        self.linear = nn.Linear(interm_dim//4 * len(self.seq_linears), 1)
        
        
    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.gaps[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.linears[i](out)
            outs.append(out)
        
        out = self.linear(torch.cat(outs, 1))
        return out
'''



'''nn.Linear(in_feat, in_feat), nn.BatchNorm1d(in_feat), nn.ReLU(), 
                nn.Linear(in_feat, in_feat//2), nn.BatchNorm1d(in_feat//2), nn.ReLU(),
                nn.Linear(in_feat//2, in_feat//2), nn.BatchNorm1d(in_feat//2), nn.ReLU(),
                nn.Linear(in_feat//2, in_feat//4), nn.BatchNorm1d(in_feat//4), nn.ReLU(),
                nn.Linear(in_feat//4, in_feat//4), nn.BatchNorm1d(in_feat//4), nn.ReLU(),'''

'''class Module_MLP(nn.Module):
    def __init__(self, interm_dim):
        super(Module_MLP, self).__init__()       
        
        in_feat = interm_dim + 1 
        
        self.seq_linears = [
            nn.Sequential(
                nn.Linear(in_feat, in_feat//4), nn.ReLU(),
            ) for _ in range(4)
        ]
        
        self.linears = nn.ModuleList(self.seq_linears)
        self.linear = nn.Linear(in_feat//4 * len(self.seq_linears), 1)


    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.linears[i](features[i])
            outs.append(out)
        
        out = self.linear(torch.cat(outs, 1))
        return out'''
        
        
class Module_MLP(nn.Module):
    def __init__(self, gtg_iter, n_classes):
        super(Module_MLP, self).__init__()
        
        in_feat = gtg_iter * n_classes # -> [128 , 10*10] = [128, 100] 
        
        self.seq_linears = nn.Sequential(
            nn.Linear(in_feat, in_feat//2), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_feat//2, in_feat//4), nn.ReLU(),
        )
        
        
        #self.linears = nn.ModuleList(self.seq_linears)
        #self.linears = nn.ModuleList(self.seq_linears)
        self.linear = nn.Linear(in_feat//4 * 4, 1)
        #self.linear = nn.Linear(in_feat * len(self.seq_linears), 1)


    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = features[i].view(features[i].size(0), -1)
            #out = self.linears[i](out)
            out = self.seq_linears(out)
            outs.append(out)
        
        out = self.linear(torch.cat(outs, 1))
        return out


    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = features[i].view(features[i].size(0), -1)
            #out = self.linears[i](out)
            out = self.seq_linears(out)
            outs.append(out)
        
        out = self.linear(torch.cat(outs, 1))
        return out
    
    
    
    
class MLP(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(MLP, self).__init__()

        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

        self.seq_linears = []

        for n_c, e_d in zip(num_channels, feature_sizes):
            size = n_c * e_d * e_d
            '''self.seq_linears.append(nn.Sequential(
                nn.Linear(size, size//10), nn.BatchNorm1d(size//10), nn.ReLU(),#65536, 32768, 16384, 8192 
                nn.Linear(size//10, interm_dim), nn.BatchNorm1d(interm_dim), nn.ReLU()
            ))'''
            self.seq_linears.append(nn.Sequential(
                nn.Linear(size, interm_dim), nn.BatchNorm1d(interm_dim), nn.ReLU()
            ))

        self.linears_1 = nn.ModuleList(self.seq_linears)
        size = interm_dim * len(self.seq_linears)
        '''self.linears_2 = nn.Sequential( #262144
            nn.Linear(size, size//2), nn.BatchNorm1d(size//2), nn.ReLU(),
            nn.Linear(size//2, size//4), nn.BatchNorm1d(size//4), nn.ReLU(),
            nn.Linear(size//4, 1),
        )'''
        self.linear = nn.Linear(interm_dim * len(self.seq_linears), 1)
        
        
    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = features[i].view(features[i].size(0), -1)
            out = self.linears_1[i](out)
            outs.append(out)
        
        #out = self.linears_2(torch.cat(outs, 1))
        out = self.linear(torch.cat(outs, 1))
        return out


class MLP_embedds(nn.Module):
    def __init__(self, embedd_dim):
        super(MLP_embedds, self).__init__()
        
        self.linears = nn.Sequential(
            nn.Linear(embedd_dim, embedd_dim//2), nn.BatchNorm1d(embedd_dim//2), nn.ReLU(),
            nn.Linear(embedd_dim//2, embedd_dim//4), nn.BatchNorm1d(embedd_dim//4), nn.ReLU(),
            nn.Linear(embedd_dim//4, embedd_dim//8), nn.BatchNorm1d(embedd_dim//8), nn.ReLU(),
            nn.Linear(embedd_dim//8, 1),
        )
        
    def forward(self, embedds):
        return self.linears(embedds)