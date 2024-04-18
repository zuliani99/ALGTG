import torch
import torch.nn as nn

from utils import init_weights_apply


class VGG(nn.Module):

    def __init__(self, features: nn.Sequential, n_classes: int):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )
        
        self.apply(init_weights_apply)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def make_layers(n_channels: int, cfg, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(n_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            n_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg, batch_norm, n_channels, n_classes):
    return VGG(make_layers(n_channels, cfgs[cfg], batch_norm=batch_norm), n_classes)


def VGG11(n_channels, n_classes): return _vgg('A', False, n_channels, n_classes)

# we will use this version
def VGG11_bn(n_channels, n_classes): return _vgg('A', True, n_channels, n_classes)

def VGG13(n_channels, n_classes): return _vgg('B', False, n_channels, n_classes)
def VGG13_bn(n_channels, n_classes): return _vgg('B', True, n_channels, n_classes)
def VGG16(n_channels, n_classes): return _vgg('D', False, n_channels, n_classes)
def VGG16_bn(n_channels, n_classes): return _vgg('D', True, n_channels, n_classes)
def VGG19(n_channels, n_classes): return _vgg('E', False, n_channels, n_classes)
def VGG19_bn(n_channels, n_classes): return _vgg('E', True, n_channels, n_classes)