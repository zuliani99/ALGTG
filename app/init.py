
from models.modules.GTG_Cls import GTG_Module
from models.modules.LossNet import LossNet
from models.backbones.ssd_pytorch.SSD import SSD, build_ssd
from models.backbones.ResNet18 import ResNet, ResNet18
from models.backbones.VGG import VGG, VGG16_bn

from datasets_creation.Classification import Cls_Datasets
from datasets_creation.Detection import Det_Dataset

from config import voc_config
from typing import Dict, Any, List, Tuple



def get_dataset(task: str, dataset_name: str, init_lab_obs: int) -> Cls_Datasets | Det_Dataset:
    if task == 'clf': return Cls_Datasets(dataset_name, init_lab_obs=init_lab_obs)
    else: return Det_Dataset(dataset_name, init_lab_obs=init_lab_obs)

    
def get_backbone(n_classes: int, n_channels: int, bbone: str) -> ResNet | VGG | SSD:
    if bbone == 'ResNet': return ResNet18(n_classes=n_classes, n_channels=n_channels)
    elif bbone == 'VGG': return VGG16_bn(n_classes=n_classes,  n_channels=n_channels)
    else: return build_ssd('train', voc_config, num_classes=n_classes)
    

def get_module(module: str, module_params: Dict[str, Any] | Tuple[Dict[str, Any], Dict[str, Any]]) -> LossNet | GTG_Module:
    if module == 'LL': return LossNet(module_params)
    else: return GTG_Module(module_params)


def get_ll_module_params(task: str, image_size: int, dataset_name: str) -> Dict[str, int | str | List[int]]:
    if task == 'detection':
        return dict(
                    feature_sizes=[512, 1024, 512, 256, 256, 256],
                    num_channels=[512, 1024, 512, 256, 256, 256],
                    interm_dim=128, task='detection'
                )
    else:
        if dataset_name != 'caltech256':   
            return dict(
                    feature_sizes=[image_size, image_size // 2, image_size // 4, image_size // 8],
                    num_channels=[64, 128, 256, 512],
                    interm_dim=128, task='cls'
                )
        else: 
            return dict(
                    feature_sizes=[224, 224, 112, 56],
                    num_channels=[64, 64, 128, 256],
                    interm_dim=128, task='cls'
                )