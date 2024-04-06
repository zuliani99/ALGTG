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
'''class LossPredLoss(nn.Module):

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
                        
        return 2 * loss'''
    
class LossPredLoss(nn.Module):
    def __init__(self,device: torch.device, margin=1):
        super(LossPredLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.reduction = 'mean'
    
    def forward(self, input, target):
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape
        
        #logger.info(f'{input.shape} - {target.shape}')

        input = (input - input.flip(0))[:len(input) // 2]
        # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors
        one = one.to(self.device)
        
        if self.reduction == 'mean':
            loss = torch.sum(torch.clamp(self.margin - one * input, min=0))
            loss = loss / input.size(0)  # Note that the size of input is already halved
        elif self.reduction == 'none':
            loss = torch.clamp(self.margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss
        


# Loss Prediction Network
class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], 
                 num_channels=[64, 128, 256, 512],
                 interm_dim=128, task='clf'):
        super(LossNet, self).__init__()

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