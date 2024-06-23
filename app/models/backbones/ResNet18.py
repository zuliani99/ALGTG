'''ResNet in PyTorch.

Reference:
https://github.com/kuangliu/pytorch-cifar
'''

import torch
import torch.nn as nn
from typing import List, Tuple
import torch.nn.functional as F

from utils import init_weights_apply


'''class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_channels=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, embedding=False):
        if embedding: # -> performed for alphamix strategy
            embedds = x
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            self.features = [out1, out2, out3, out4]
            
            out = F.avg_pool2d(out4, 4)
            embedds = out.view(out.size(0), -1)
        
        out = self.linear(embedds)
        
        return out, embedds
    

    '''def forward(self, x: torch.Tensor, embedding = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if embedding: # -> performed for alphamix strategy
            embedds = x
        else:
            out = self.conv1(x)
            
            out1 = self.conv2_x(out)
            out2 = self.conv3_x(out1)
            out3 = self.conv4_x(out2)
            out4 = self.conv5_x(out3)
            
            self.features = [out1, out2, out3, out4]
            out = self.avg_pool(out4)

            embedds = out.view(out.size(0), -1)
                        
        out = self.classifier(embedds)

        return out, embedds'''
    
    def get_rich_features_shape(self) -> List[int]:
        return [64, 128, 256, 512]
    
    def get_features(self) -> List[torch.Tensor]:
        return self.features
    
    def get_embedding_dim(self) -> int:
        return self.linear.in_features


'''class ResNet(nn.Module):
        
    def __init__(self, block: BasicBlock, num_block: List[int], n_channels=3, num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        
        #self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
            
        self.apply(init_weights_apply)
        

    def _make_layer(self, block: BasicBlock, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, planes, stride))
            self.in_channels = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, embedding = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if embedding: # -> performed for alphamix strategy
            embedds = x
        else:
            out = self.conv1(x)
            
            out1 = self.conv2_x(out)
            out2 = self.conv3_x(out1)
            out3 = self.conv4_x(out2)
            out4 = self.conv5_x(out3)
            
            self.features = [out1, out2, out3, out4]
            out = self.avg_pool(out4)

            embedds = out.view(out.size(0), -1)
                        
        out = self.classifier(embedds)

        return out, embedds
    
    def get_rich_features_shape(self) -> List[int]:
        return [64, 128, 256, 512]
    
    def get_features(self) -> List[torch.Tensor]:
        return self.features
    
    def get_embedding_dim(self) -> int:
        return self.classifier.in_features'''


def ResNet18(n_classes=10, n_channels=3) -> ResNet:
    return ResNet(BasicBlock, [2,2,2,2], n_channels, n_classes) # type: ignore
