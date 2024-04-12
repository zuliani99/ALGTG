
import torch
from models.GTG_Cls import Class_GTG
from models.ResNet18 import ResNet_LL
from models.ssd_pytorch.SSD import SSD_LL
from datasets_creation.Classification import Cls_Datasets
from datasets_creation.Detection import Det_Dataset
from config import voc_config
from typing import Dict, Any


def get_dataset(task, dataset_name: str, init_lab_obs: int) -> Cls_Datasets | Det_Dataset:
    if task == 'clf': return Cls_Datasets(dataset_name, init_lab_obs=init_lab_obs)
    else: return Det_Dataset(dataset_name, init_lab_obs=init_lab_obs)


def get_model(image_size: int, n_classes: int, n_channels: int, device: torch.device, task: str) -> ResNet_LL | SSD_LL:
    if task == 'clf': 
        resnet_ll = ResNet_LL(image_size, n_classes=n_classes,  n_channels=n_channels).to(device)
        return resnet_ll
    else: 
        loss_net_params = dict(
            feature_sizes=[512, 1024, 512, 256, 256, 256],
            num_channels=[512, 1024, 512, 256, 256, 256],
            task='detection'
        )
        ssd_ll = SSD_LL('train', voc_config, num_classes=n_classes, ln_p=loss_net_params).to(device)
        return ssd_ll
    

def get_gtg_module(gtg_p: Dict[str, Any], n_top_k_obs: int, n_classes: int, init_lab_obs: int, image_size: int, n_channels: int, device: torch.device):
    gtg_module_params = dict(
        **gtg_p, n_top_k_obs=n_top_k_obs, n_classes=n_classes, init_lab_obs=init_lab_obs, device=device
    )
    resnet_params = dict( image_size=image_size, n_channels=n_channels, device=device )
    gtg_cls = Class_GTG(gtg_module_params, resnet_params).to(device)
    
    return gtg_cls