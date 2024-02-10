
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    
    
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        embeds = out.view(out.size(0), -1)
        out = self.linear(embeds)
        return out, embeds, None, None

    


class ResNet_Weird(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Weird, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
        
        self.global_average_pooling_1 = nn.AvgPool2d(32)        
        self.global_average_pooling_2 = nn.AvgPool2d(16)
        self.global_average_pooling_3 = nn.AvgPool2d(8)
        self.global_average_pooling_4 = nn.AvgPool2d(4)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    
    def forward(self, x):
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
        
        out = F.avg_pool2d(out, 4)
        embeds = out.view(out.size(0), -1)
        out = self.linear(embeds)
        concat = torch.cat((fc1, fc2, fc3, fc4), dim=1)
        out_weird = self.fc_concat(concat)
        return out, embeds, out_weird, concat




class LearningLoss(nn.Module):

    def __init__(self, device, margin=1):
        super(LearningLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, output, real_loss):
        output = output.squeeze()
        
        mid = output.shape[0] // 2
        
        output_1 = output[:mid]
        real_loss_1 = real_loss[:mid]
        
        output_2 = output[mid:]
        real_loss_2 = real_loss[mid:]
        
        plus_minus_1 = (real_loss_1 > real_loss_2)
        plus_minus_1 = plus_minus_1.type(torch.FloatTensor)
        
        check_zeros = (plus_minus_1 == 0)
        check_zeros = check_zeros.type(torch.FloatTensor)
        
        plus_minus_1_final = plus_minus_1 - check_zeros
        plus_minus_1_final = plus_minus_1_final.to(self.device)
        calc_loss = (-1) * plus_minus_1_final * (output_1 - output_2) + self.margin
        greater_than_zero = calc_loss > 0
        greater_than_zero = greater_than_zero.type(torch.FloatTensor)
        greater_than_zero = greater_than_zero.to(self.device)
        loss = calc_loss * greater_than_zero
        return torch.mean(loss)


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss



# Define the ResNet18 model
class OriginalResNet18(nn.Module):
    def __init__(self, num_classes):
        super(OriginalResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    
    def forward(self, x):
        return self.resnet18(x)