
import torch
from models.ResNet18 import ResNet_LL
from models.ssd_pytorch.SSD import SSD_LL
from datasets_creation.Classification import Cls_Datasets
from datasets_creation.Detection import Det_Dataset
from config import voc_config


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
    

