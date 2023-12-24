'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_utils
import genereate_splits as gs
import numpy as np
from visual import plotter

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

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
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        fc7 = out.view(out.size(0), -1)
        out = self.linear(fc7)
        return out, fc7


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
        self.fc_weird_1 = nn.Linear(128, 128)
        self.fc_weird_2 = nn.Linear(256, 128)
        self.fc_weird_3 = nn.Linear(512, 128)
        self.fc_concat = nn.Linear(128 * 3, 1)
        self.global_average_pooling_1 = nn.AvgPool2d(16)
        self.global_average_pooling_2 = nn.AvgPool2d(8)
        self.global_average_pooling_3 = nn.AvgPool2d(4)


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
        out = self.layer2(out)
        weird_pool_1 = self.global_average_pooling_1(out)
        fc1 = weird_pool_1.view(weird_pool_1.size(0), -1)
        fc1 = self.fc_weird_1(fc1)
        out = self.layer3(out)
        weird_pool_2 = self.global_average_pooling_2(out)
        fc2 = weird_pool_2.view(weird_pool_2.size(0), -1)
        fc2 = self.fc_weird_2(fc2)
        out = self.layer4(out)
        weird_pool_3 = self.global_average_pooling_3(out)
        fc3 = weird_pool_3.view(weird_pool_3.size(0), -1)
        fc3 = self.fc_weird_3(fc3)
        out = F.avg_pool2d(out, 4)
        fc7 = out.view(out.size(0), -1)
        out = self.linear(fc7)
        concat = torch.cat((fc1, fc2, fc3), dim=1)
        out_weird = self.fc_concat(concat)
        return out, fc7, out_weird, concat


class LearningLoss(nn.Module):
    """
    """

    def __init__(self, margin=1):
        super(LearningLoss, self).__init__()
        self.margin = 1

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
        plus_minus_1_final = plus_minus_1_final.cuda()
        calc_loss = (-1) * plus_minus_1_final * (output_1 - output_2) + self.margin
        greater_than_zero = calc_loss > 0
        greater_than_zero = greater_than_zero.type(torch.FloatTensor)
        greater_than_zero = greater_than_zero.cuda()
        loss = calc_loss * greater_than_zero
        return torch.mean(loss)


def ResNet_LLAL(num_classes=10):
    return ResNet_Weird(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet18(num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

plotter = plotter.VisdomLinePlotter(env_name='resnet-laal')


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_sample = 1000

    train_dset, test_dset, num_classes = gs.get_dataset('cifar10', padding=2)

    net = ResNet_Weird(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)
    # net = ResNet18().to(device)

    idx = np.arange(50000)
    np.random.shuffle(idx)
    idx = idx[:num_sample]
    train_loader, _, _ = gs.get_loader(train_dset, gs.get_sampler(idx), batch_size=128)

    new_mean, new_std = gs.online_mean_and_sd(train_loader)
    train_dset, test_dset, num_classes = gs.get_dataset('cifar10', padding=2, mean=new_mean, std=new_std)
    train_loader, _, _ = gs.get_loader(train_dset, gs.get_sampler(idx), batch_size=128)
    te_ids = np.arange(10000)
    test_loader, test_dict, test_dict_inv = gs.get_loader(test_dset, gs.get_sampler(te_ids), batch_size=128)

    loss_weird = LearningLoss()
    cross_entropy = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)


    def eval(net, loader, device):

        use_LLAL = True

        net.eval()
        correct = 0
        total = 0
        for k, data in enumerate(loader):
            inputs, labels, indices = data
            inputs, labels = inputs.to(device), labels.to(device)

            if use_LLAL:
                outputs, _, loss, _ = net(inputs)
            else:
                outputs, _ = net(inputs)

            _, outputs = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (outputs == labels).sum()

        accuracy = correct.data.cpu().item() / total

        return accuracy

    def train(net, train_loader, test_loader, l_ce, l_weird, optimizer, device):
        # consider calling eval a the end of each epoch and then saving the best performing net

        use_LLAL = True

        accuracies = []
        weight = 1.   # 120 = 0
        for e in range(200):
            net.train()
            if e > 120:
                weight = 0
            loss_ce_total = 0
            loss_weird_total = 0
            for k, data in enumerate(train_loader):
                inputs, labels, indices = data

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                if use_LLAL:
                    outputs, _, out_weird, _ = net(inputs)
                    loss_ce = l_ce(outputs, labels)
                    loss_weird = l_weird(out_weird, loss_ce)
                    loss_ce = torch.mean(loss_ce)
                    loss = loss_ce + weight * loss_weird
                    loss_ce_total += loss_ce
                    loss_weird_total += loss_weird
                else:
                    outputs, _ = net(inputs)
                    loss_ce = l_ce(outputs, labels)
                    loss_ce = torch.mean(loss_ce)
                    loss = loss_ce  # comment to LLAL loss
                    loss_ce_total += loss_ce

                loss.backward()
                optimizer.step()

            acc = eval(net, test_loader, device)
            print("Epoch " + str(e) + ": " + str(acc))
            plotter.plot('loss', 'Cross_entropy', 'Loss_' + str(num_sample), e, loss_ce_total.item() / k)
            plotter.plot('loss', 'LAAL', 'Loss_' + str(num_sample), e, loss_weird_total.item() / k)
            plotter.plot('loss', 'Accuracy', 'Accuracy_' + str(num_sample), e, acc)


            print("\n")
            accuracies.append(acc)

            if e == 160:
                for g in optimizer.param_groups:
                    g['lr'] = 0.01

        print("Best accuracy is: " + str(max(accuracies)))



    train(net, train_loader, test_loader, cross_entropy, loss_weird, optimizer, device)


# test()