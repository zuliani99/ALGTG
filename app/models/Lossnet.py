'''
Reference:
    https://github.com/euphoria0-0/Learning-Loss-for-Active-Learning/
'''

import torch.nn as nn
import torch.nn.functional as F
import torch


import logging
logger = logging.getLogger(__name__)
    
# Loss Prediction Loss
class LossPredLoss(nn.Module):

    def __init__(self, device: torch.device, margin=1):
        super(LossPredLoss, self).__init__()
        self.margin = margin
        self.device = device
        
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        assert len(inputs) % 2 == 0, 'The batch size is not even.'
        assert inputs.shape == inputs.flip(0).shape
                
        mid = inputs.shape[0] // 2

        pred_lossi = inputs[:mid].squeeze()
        target_lossi = targets[:mid]

        pred_lossj = inputs[mid:].squeeze()
        target_lossj = targets[mid:]
        
        final_target = torch.sign(target_lossi - target_lossj).to(self.device)
        
        loss = F.margin_ranking_loss(pred_lossi, pred_lossj, final_target, margin=self.margin)
                        
        return 2 * loss        
    


# Loss Prediction Network
class LossNet(nn.Module):
    def __init__(self, dict_params = {}):
        super(LossNet, self).__init__()

        feature_sizes = dict_params['feature_sizes'] if 'feature_sizes' in dict_params else [32, 16, 8, 4] 
        num_channels = dict_params['num_channels'] if 'num_channels' in dict_params else [64, 128, 256, 512]
        interm_dim = dict_params['interm_dim'] if 'interm_dim' in dict_params else 128
        task = dict_params['task'] if 'task' in dict_params else 'clf'
        
        self.GAP = []
        for feature_size in feature_sizes:
            if task == 'detection': self.GAP.append(nn.AdaptiveAvgPool2d((1, 1)))
            else: self.GAP.append(nn.AvgPool2d(feature_size))
            
        self.GAP = nn.ModuleList(self.GAP)

        self.FC = []
        for num_channel in num_channels: self.FC.append(nn.Linear(num_channel, interm_dim))
        self.FC = nn.ModuleList(self.FC)

        self.linear = nn.Linear(len(num_channels) * interm_dim, 1)

    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.GAP[i](features[i])
            out = out.view(out.size(0), -1)
            out = F.relu(self.FC[i](out))
            outs.append(out)

        out = self.linear(torch.cat(outs, 1))
        return out
     