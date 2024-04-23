'''
Reference:
    https://github.com/euphoria0-0/Learning-Loss-for-Active-Learning/
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import init_weights_apply

import logging
logger = logging.getLogger(__name__)
    
    
# Loss Prediction Loss
class LossPredLoss(nn.Module):
    def __init__(self, device, margin=1.0, reduction='mean'):
        super(LossPredLoss, self).__init__()

        self.device = device
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)

    def forward(self, prediction, target):
        batch_half = target.shape[0]//2
        
        target_ranking = (target[:batch_half]-target[batch_half:]).detach()
        target_ranking = 2 * torch.sign(torch.clamp(target_ranking, min=0)) - 1
        
        predictions_1 = prediction[:batch_half].squeeze()
        predictions_2 = prediction[batch_half:].squeeze()
        
        loss = self.margin_ranking_loss(predictions_1, predictions_2, target_ranking)
        
        return loss


# Loss Prediction Network
class LossNet(nn.Module):
    def __init__(self, LL_params):
        super(LossNet, self).__init__()

        feature_sizes = LL_params['feature_sizes']
        num_channels = LL_params['num_channels']
        interm_dim = LL_params['interm_dim']
        task = LL_params['task']
        
        self.GAP = []
        for feature_size in feature_sizes:
            if task == 'detection': self.GAP.append(nn.AdaptiveAvgPool2d((1, 1)))
            else: self.GAP.append(nn.AvgPool2d(feature_size))
            
        self.GAP = nn.ModuleList(self.GAP)

        self.FC = []
        for num_channel in num_channels: self.FC.append(nn.Linear(num_channel, interm_dim))
        self.FC = nn.ModuleList(self.FC)

        self.linear = nn.Linear(len(num_channels) * interm_dim, 1)
        
        self.apply(init_weights_apply)

    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.GAP[i](features[i])
            out = out.view(out.size(0), -1)
            out = F.relu(self.FC[i](out))
            outs.append(out)

        out = self.linear(torch.cat(outs, 1))
        return out
        