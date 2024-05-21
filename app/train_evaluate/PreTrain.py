
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

import logging
logger = logging.getLogger(__name__)

    
    
class PreTrain_BackBone(nn.Module):
    def __init__(self, backbone):
        super(PreTrain_BackBone, self).__init__()
        self.backbone = backbone
        self.mlp = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        
    def forward(self, x):
        _, embedds = self.backbone(x)
        out = self.mlp(embedds)
        return out
    
    
class BinaryDataset(Dataset):
    def __init__(self, lab_subset, unlab_subset):
        self.lab_subset = lab_subset
        self.unlab_subset = unlab_subset
        
    def __len__(self):
        return len(self.lab_subset) + len(self.unlab_subset)
    
    def __getitem__(self, idx): 
        data = self.lab_subset[self.lab_subset.indices.index(idx)] if idx in self.lab_subset.indices else self.unlab_subset[self.unlab_subset.indices.index(idx)]
        return data[1], 1. if idx in self.lab_subset.indices else 0.
        

class PreTrain:
    def __init__(self, device, lab_subset, unlab_subset, backbone) -> None:
        self.device = device
        self.train_dl = DataLoader(BinaryDataset(lab_subset, unlab_subset), batch_size=128, shuffle=True)
        self.epochs = 10
        self.pt_bb = PreTrain_BackBone(backbone).to(device)
    
    def train(self):
        optimizer = Adam(self.pt_bb.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        loss = 0.

        for epoch in range(self.epochs):
            
            for images, labels in self.train_dl:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                y_pred = self.pt_bb(images).squeeze().to(torch.float64)
                loss = criterion(y_pred, labels)
                loss.backward()
                optimizer.step()
                
                loss += loss.item()
            
            loss /= len(self.train_dl)
            
            logger.info(f' Epoch: {epoch} | loss -> {loss}')
        
        return self.pt_bb.backbone.state_dict()
    