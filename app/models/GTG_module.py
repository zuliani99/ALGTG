
import torch
import torch.nn as nn

class GTG_Module(nn.Module):
    def __init__(self):
        super(GTG_Module, self).__init__()

    
    def forward(self, x):
        return x