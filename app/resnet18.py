import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

# Define the ResNet18 model
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.change_weigths(list(self.resnet18.children()))

    def change_weigths(self, childs):
        if isinstance(childs, nn.Conv2d):
            init.xavier_uniform_(childs.weight, gain=nn.init.calculate_gain('relu'))
            if childs.bias is not None:
                init.constant_(childs.bias, 0)
        else:
          for child in childs:
            if (isinstance(child, nn.Sequential) or isinstance(child, models.resnet.BasicBlock)):
                  self.change_weigths(list(child.children()))
            else:
                if isinstance(child, nn.Conv2d):
                  init.xavier_uniform_(child.weight, gain=nn.init.calculate_gain('relu'))
                  if child.bias is not None:
                      init.constant_(child.bias, 0)
    
    def forward(self, x):
        return self.resnet18(x)