'''
import torch
import torch.nn as nn
import torch.nn.init as init

from models.ResNet18 import ResNet_LL
from models.ssd_pytorch.SSD import SSD_LL
from config import voc_config'''


'''def get_ssd_model(n_classes: int, device: torch.device) -> SSD:
    
    loss_net = LossNet(feature_sizes=[512, 1024, 512, 256, 256, 256],
                       num_channels=[512, 1024, 512, 256, 256, 256],
                       task='detection')
    
    loss_net.apply(init_weights_apply)
    loss_net = loss_net.to(device)
    

    ssd_net = build_ssd(loss_net, 'train', voc_config, num_classes=n_classes)
    vgg_weights = torch.load('app/models/ssd_pytorch/vgg16_reducedfc.pth')
    ssd_net.vgg.load_state_dict(vgg_weights)
    # initialize
    ssd_net.extras.apply(init_weights_apply)
    ssd_net.loc.apply(init_weights_apply)
    ssd_net.conf.apply(init_weights_apply)
    ssd_net = ssd_net.to(device)
    
    return ssd_net


def get_resnet_model(image_size: int, n_classes: int, n_channels: int, device: torch.device) -> ResNet:
    
    
    loss_net = LossNet()
    loss_net.apply(init_weights_apply)
    loss_net = loss_net.to(device)
    
    resnet = ResNet18(image_size, loss_net, n_classes=n_classes, n_channels=n_channels)
    resnet.apply(init_weights_apply)
    resnet = resnet.to(device)

    return resnet'''
    
'''# weights initiaization
def init_weights_apply(m: torch.nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=1e-3)
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    
    
def get_ssd_model(n_classes: int, device: torch.device) -> SSD_LL:
    loss_net_params = dict(
        feature_sizes=[512, 1024, 512, 256, 256, 256],
        num_channels=[512, 1024, 512, 256, 256, 256],
        task='detection'
    )
    ssd_ll = SSD_LL('train', voc_config, num_classes=n_classes, ln_p=loss_net_params)
    ssd_ll.apply(init_weights_apply)
    ssd_ll = ssd_ll.to(device)
    return ssd_ll


def get_resnet_model(image_size: int, n_classes: int, n_channels: int, device: torch.device) -> ResNet_LL:
    resnet_ll = ResNet_LL(image_size, n_classes=n_classes,  n_channels=n_channels)
    resnet_ll.apply(init_weights_apply)
    resnet_ll = resnet_ll.to(device)
    return resnet_ll'''