    
import torch
from config import voc_config
from models.ResNet18 import ResNet_LL
from models.ssd_pytorch.SSD import SSD_LL
from utils import init_weights_apply


def get_ssd_model(n_classes: int, device: torch.device) -> SSD_LL:
    loss_net_params = dict(
        feature_sizes=[512, 1024, 512, 256, 256, 256],
        num_channels=[512, 1024, 512, 256, 256, 256],
        task='detection'
    )
    ssd_ll = SSD_LL('train', voc_config, num_classes=n_classes, ln_p=loss_net_params).to(device)
    ssd_ll.backbone.apply(init_weights_apply)
    ssd_ll.loss_net.apply(init_weights_apply)
    
    ssd_ll.backbone.extras.apply(init_weights_apply)
    ssd_ll.backbone.loc.apply(init_weights_apply)
    ssd_ll.backbone.conf.apply(init_weights_apply)
    
    return ssd_ll


def get_resnet_model(image_size: int, n_classes: int, n_channels: int, device: torch.device) -> ResNet_LL:
    resnet_ll = ResNet_LL(image_size, n_classes=n_classes, n_channels=n_channels).to(device)
    resnet_ll.backbone.apply(init_weights_apply)
    resnet_ll.loss_net.apply(init_weights_apply)
    return resnet_ll