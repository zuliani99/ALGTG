
import torch

from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler


from models.backbones.ResNet18 import ResNet18
from train_evaluate.PreTrain import PreTrain
from utils import set_seeds
from datasets_creation.Detection import detection_collate
from .Workers.Cls_TrainWorker import Cls_TrainWorker
from .Workers.Det_TrainWorker import Det_TrainWorker

from typing import Tuple, Dict, Any, List

    
    
def train(params: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    set_seeds()
    
    ct_p = params["ct_p"]
    t_p = params["t_p"]   
    
    ct_p["Master_Model"] = ct_p["Master_Model"].to(params["ct_p"]["device"])
    batch_size = t_p[ct_p["dataset_name"]]["batch_size"][params["module_name_only"]]
    params["batch_size"] = batch_size
    
    dict_dl = dict(batch_size=batch_size, pin_memory=True)
    
    if ct_p["task"] == 'detection': 
        dict_dl["collate_fn"] = detection_collate
        
        t_p["epoch_size"] = len(params["labelled_subset"]) // batch_size
        t_p["max_iter"] = t_p["epochs"] * t_p["epoch_size"]
        
        TrainWorker = Det_TrainWorker
    
    else: TrainWorker = Cls_TrainWorker
    
    if params["module_name_only"] == 'GTGModule':
        train_dl = (
            Subset(ct_p["Dataset"].train_ds, params["labelled_indices"]),
            Subset(ct_p["Dataset"].train_ds, params["rand_unlab_sample"])
        )
    else: 
        train_dl = DataLoader(ct_p["Dataset"].train_ds, sampler=SubsetRandomSampler(params["labelled_indices"]), **dict_dl)
        
    params["train_dl"] = train_dl
    params["test_dl"] = DataLoader(ct_p["Dataset"].test_ds, shuffle=False, **dict_dl)
    
    train_test = TrainWorker(params["ct_p"]["device"], params)
    
    if ct_p["bbone_pre_train"]: pre_train(params["ct_p"]["device"], params)
        
    train_results = train_test.train().cpu().tolist()
    test_results = train_test.test().cpu().tolist()
    
    return train_results, test_results


def pre_train(device, params):
    # Pretrain our backbone via Binary classification task (labelled, unlabelled)
    pt = PreTrain(
        device=device, backbone=ResNet18(),
        # I ahve to uise the labelled dataset since I need to combine both labeleld and unlabeleld indices to create the dataloader for the binary classification task
        lab_subset=Subset(params["ct_p"]["Dataset"].lab_train_ds, params["labelled_indices"]),
        unlab_subset=Subset(params["ct_p"]["Dataset"].lab_train_ds, params["unlabelled_indices"])
    )
    
    torch.save(dict(state_dict = pt.train()), f'app/checkpoints/{params["Dataset"]["dataset_name"]}/pratrained_BB.pth.tar')
