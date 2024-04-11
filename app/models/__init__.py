'''    
import torch
from app.models.GTG_Cls import Class_GTG
from config import voc_config
from models.ResNet18 import ResNet_LL
from models.ssd_pytorch.SSD import SSD_LL
from utils import init_weights_apply
from typing import Dict, Any


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


def get_gtg_module(gtg_p: Dict[str, Any], n_top_k_obs: int, n_classes: int, init_lab_obs: int, image_size: int, n_channels: int, device: torch.device):
    gtg_module_params = dict(
        **gtg_p, n_top_k_obs=n_top_k_obs, n_classes=n_classes, init_lab_obs=init_lab_obs, device=device
    )
    resnet_params = dict( image_size=image_size, n_channels=n_channels, device=device )
    gtg_cls = Class_GTG(gtg_module_params, resnet_params).to(device)
    
    gtg_cls.backbone.apply(init_weights_apply)
    gtg_cls.gtg.apply(init_weights_apply)
    
    return gtg_cls'''