
from models.modules.GTG_Cls import GTG_Module
from models.modules.LossNet import LossNet
from models.backbones.ssd_pytorch.SSD import SSD, build_ssd
from models.backbones.ResNet18 import ResNet, ResNet18

from datasets_creation.Classification import Cls_Datasets
from datasets_creation.Detection import Det_Dataset

from config import voc_config
from typing import Dict, Any, List, Tuple


def get_dataset(task, dataset_name: str, init_lab_obs: int) -> Cls_Datasets | Det_Dataset:
    if task == 'clf': return Cls_Datasets(dataset_name, init_lab_obs=init_lab_obs)
    else: return Det_Dataset(dataset_name, init_lab_obs=init_lab_obs)

    
def get_backbone(n_classes: int, n_channels: int, task: str) -> ResNet | SSD:
    if task == 'clf': return ResNet18(n_classes=n_classes,  n_channels=n_channels)
    else: return build_ssd('train', voc_config, num_classes=n_classes)
    

def get_module(module: str, module_params: Dict[str, Any] | Tuple[Dict[str, Any], Dict[str, Any]]) -> LossNet | GTG_Module:
    if module == 'LL': return LossNet(module_params)
    else: return GTG_Module(module_params)


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