'''ResNet in PyTorch.

Reference:
https://github.com/kuangliu/pytorch-cifar
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from utils import init_weights_apply


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride=1) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, image_size: int, block: BasicBlock, num_blocks: List[int], n_channels=3, n_classes=10) -> None:
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=image_size // 32, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, n_classes)
        
        self.apply(init_weights_apply)

    def _make_layer(self, block: BasicBlock, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def get_rich_features_shape(self) -> List[int]:
        return [64, 128, 256, 512]
    
    def get_features(self) -> List[torch.Tensor]:
        return self.features
    
    def get_embedding_dim(self) -> int:
        return self.linear.in_features


def ResNet18(image_size: int, n_classes=10, n_channels=3) -> ResNet:
    return ResNet(image_size, BasicBlock, [2,2,2,2], n_channels, n_classes) # type: ignore
