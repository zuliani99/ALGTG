
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models


# Define the ResNet18 model
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    
    def forward(self, x):
        return self.resnet18(x)