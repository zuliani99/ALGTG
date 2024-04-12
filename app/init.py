
import torch
#from app.models.GTG_Cls import Class_GTG
#from models.ssd_pytorch.SSD import SSD_LL

from app.models.modules.GTG_Cls import GTG_Module
from app.models.modules.LossNet import LossNet
from app.utils import init_weights_apply
from models.backbones.ssd_pytorch.SSD import SSD, build_ssd
from models.backbones.ResNet18 import ResNet, ResNet18

from datasets_creation.Classification import Cls_Datasets
from datasets_creation.Detection import Det_Dataset

from config import voc_config
from typing import Dict, Any, List


def get_dataset(task, dataset_name: str, init_lab_obs: int) -> Cls_Datasets | Det_Dataset:
    if task == 'clf': return Cls_Datasets(dataset_name, init_lab_obs=init_lab_obs)
    else: return Det_Dataset(dataset_name, init_lab_obs=init_lab_obs)


'''def get_model(image_size: int, n_classes: int, n_channels: int, device: torch.device, task: str) -> ResNet_LL | SSD_LL:
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
    
    return gtg_cls'''
    
    
def get_backbone(image_size: int, n_classes: int, n_channels: int, task: str) -> ResNet | SSD:
    if task == 'clf': 
        resnet = ResNet18(image_size, n_classes=n_classes,  n_channels=n_channels)
        resnet.apply(init_weights_apply)
        return resnet
    else: 
        ssd_net = build_ssd('train', voc_config, num_classes=n_classes)
        vgg_weights = torch.load('app/models/ssd_pytorch/vgg16_reducedfc.pth')
        ssd_net.vgg.load_state_dict(vgg_weights)
        ssd_net.extras.apply(init_weights_apply)
        ssd_net.loc.apply(init_weights_apply)
        ssd_net.conf.apply(init_weights_apply)
        return ssd_net
    

def get_module(module: str, module_params: Dict[str, Any]) -> LossNet | GTG_Module:
    if module == 'LL': 
        ll = LossNet(module_params)
        ll.apply(init_weights_apply)
        return ll
    else:
        gtg = GTG_Module(module_params)
        gtg.apply(init_weights_apply)
        return gtg


def get_ll_module_params(task: str) -> Dict[str, int | str | List[int]]:
    if task == 'detection':
        return dict(
                    feature_sizes=[512, 1024, 512, 256, 256, 256],
                    num_channels=[512, 1024, 512, 256, 256, 256],
                    interm_dim=128,
                    task='detection'
                )
    else:
        return dict(
                    feature_sizes=[32, 16, 8, 4],
                    num_channels=[64, 128, 256, 512],
                    interm_dim=128,
                    task='cls'
                )



'''def get_gtg_module(gtg_p: Dict[str, Any], n_top_k_obs: int, n_classes: int, init_lab_obs: int, image_size: int, n_channels: int, device: torch.device):
    gtg_module_params = dict(
        **gtg_p, n_top_k_obs=n_top_k_obs, n_classes=n_classes, init_lab_obs=init_lab_obs, device=device
    )
    resnet_params = dict( image_size=image_size, n_channels=n_channels, device=device )
    gtg_cls = Class_GTG(gtg_module_params, resnet_params).to(device)
    
    return gtg_cls'''
    
