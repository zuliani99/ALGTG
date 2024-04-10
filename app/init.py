
import torch
from models import get_resnet_model, get_ssd_model
from models.ResNet18 import ResNet_LL
from models.ssd_pytorch.SSD import SSD_LL
from datasets_creation.Classification import Cls_Datasets
from datasets_creation.Detection import Det_Datasets

def get_dataset(task, dataset_name: str, 
                init_lab_obs: int) -> Cls_Datasets | Det_Datasets:
    if task == 'clf': return Cls_Datasets(dataset_name, init_lab_obs=init_lab_obs)
    else: return Det_Datasets(dataset_name, init_lab_obs=init_lab_obs)


def get_model(image_size: int, n_classes: int, n_channels: int, 
              device: torch.device, task: str) -> ResNet_LL | SSD_LL:
    if task == 'clf': return get_resnet_model(image_size, n_classes, n_channels, device)
    else: return get_ssd_model(n_classes, device)