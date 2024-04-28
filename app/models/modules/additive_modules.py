
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