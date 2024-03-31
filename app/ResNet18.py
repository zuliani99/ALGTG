
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple


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
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    


class ResNet_Weird(nn.Module):
    
    def __init__(self, block: BasicBlock, num_blocks: List[int], image_size: int, num_classes=10, n_channels=3) -> None:
        super(ResNet_Weird, self).__init__()
        self.in_planes = 64
        self.image_size = image_size


        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        self.fc_weird_1 = nn.Linear(64, 128)        
        self.fc_weird_2 = nn.Linear(128, 128)
        self.fc_weird_3 = nn.Linear(256, 128)
        self.fc_weird_4 = nn.Linear(512, 128)
        self.fc_concat = nn.Linear(128 * 4, 1)
        
        self.global_average_pooling_1 = nn.AvgPool2d(image_size)
        self.global_average_pooling_2 = nn.AvgPool2d(int(image_size / 2))
        self.global_average_pooling_3 = nn.AvgPool2d(int(image_size / 4))
        self.global_average_pooling_4 = nn.AvgPool2d(int(image_size / 8))


    def _make_layer(self, block: BasicBlock, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        weird_pool_1 = self.global_average_pooling_1(out)
        fc1 = weird_pool_1.view(weird_pool_1.size(0), -1)
        fc1 = F.relu(self.fc_weird_1(fc1))        
        
        out = self.layer2(out)
        weird_pool_2 = self.global_average_pooling_2(out)
        fc2 = weird_pool_2.view(weird_pool_2.size(0), -1)
        fc2 = F.relu(self.fc_weird_2(fc2))
        
        out = self.layer3(out)
        weird_pool_3 = self.global_average_pooling_3(out)
        fc3 = weird_pool_3.view(weird_pool_3.size(0), -1)
        fc3 = F.relu(self.fc_weird_3(fc3))
        
        out = self.layer4(out)
        weird_pool_4 = self.global_average_pooling_4(out)
        fc4 = weird_pool_4.view(weird_pool_4.size(0), -1)
        fc4 = F.relu(self.fc_weird_4(fc4))
        
        out = F.avg_pool2d(out, int(self.image_size / 8))
        embeds = out.view(out.size(0), -1)
        out = self.linear(embeds)
        concat = torch.cat((fc1, fc2, fc3, fc4), dim=1)
        out_weird = self.fc_concat(concat)
        return out, embeds, out_weird, concat




class LearningLoss(nn.Module):

    def __init__(self, device: torch.device, margin=1):
        super(LearningLoss, self).__init__()
        self.margin = margin
        self.device = device
        
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        mid = inputs.shape[0] // 2

        pred_lossi = inputs[:mid].squeeze()
        target_lossi = targets[:mid]

        pred_lossj = inputs[mid:].squeeze()
        target_lossj = targets[mid:]
        
        final_target = torch.sign(target_lossi - target_lossj).to(self.device)
        
        loss = F.margin_ranking_loss(pred_lossi, pred_lossj, final_target, margin=self.margin)
                        
        return 2 * loss
