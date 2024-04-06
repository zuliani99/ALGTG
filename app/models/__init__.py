
import torch

from models.Lossnet import LossNet
from models.ResNet18 import ResNet, ResNet18
from models.ssd_pytorch.SSD import SSD, build_ssd
from config import voc_config
from utils import init_weights_apply


def get_ssd_model(n_classes: int, device: torch.device) -> SSD:
    
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


def get_resnet_model(n_classes: int, n_channels: int, device: torch.device) -> ResNet:
    
    
    loss_net = LossNet()
    loss_net.apply(init_weights_apply)
    loss_net = loss_net.to(device)
    
    resnet = ResNet18(loss_net, n_classes=n_classes, n_channels=n_channels)
    resnet.apply(init_weights_apply)
    resnet = resnet.to(device)

    return resnet