
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision import datasets
import torch

from typing import List, Tuple

import os
from utils import count_class_observation
from config import cls_datasets 

import logging
logger = logging.getLogger(__name__)



def download_tinyimagenet() -> None:
    if not os.path.exists('datasets/tiny-imagenet-200'):
        logger.info(' => Downloading Tiny-IMAGENET Dataset')
        os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        os.system('unzip tiny-imagenet-200.zip -d datasets')
        os.remove('tiny-imagenet-200.zip')
        logger.info(' DONE\n')
    else:
        logger.info('Tiny-IMAGENET Dataset already downloaded')



class Cls_Datasets():
    
    def __init__(self, dataset_name: str, init_lab_obs: int) -> None:
        
        self.transformed_trainset = Cls_Dataset(dataset_name=dataset_name, bool_train=True, bool_transform=True)
        self.non_transformed_trainset = Cls_Dataset(dataset_name=dataset_name, bool_train=True, bool_transform=False)
        
        self.test_ds = Cls_Dataset(dataset_name=dataset_name, bool_train=False, bool_transform=False)

        self.n_classes: int = cls_datasets[dataset_name]['n_classes']
        self.n_channels: int = cls_datasets[dataset_name]['channels']
        self.dataset_id: int = cls_datasets[dataset_name]['id']
        self.image_size: int = cls_datasets[dataset_name]['image_size']
        self.classes: List[str] = cls_datasets[dataset_name]['classes']
    
        self.get_initial_subsets(init_lab_obs)
    
    
    
    def get_initial_subsets(self, init_lab_obs: int) -> None:

        train_size = len(self.transformed_trainset)
        
        # random shuffle of the train indices
        shuffled_indices = torch.randperm(train_size)
        # each time should be a new shuffle, thus a new train-validation, labeled-unlabeled split

        # indices for the labeled and unlabeled sets
        self.labeled_indices: List[int] = shuffled_indices[:init_lab_obs].tolist()
        self.unlabeled_indices: List[int] = shuffled_indices[init_lab_obs:].tolist()
        
        logger.info(f' Initial subset of labeled observations composed with: {count_class_observation(
            self.classes, Subset(self.transformed_trainset, self.labeled_indices)
        )}')

    


class Cls_Dataset(Dataset):
    def __init__(self, dataset_name: str, bool_train: bool, bool_transform = True) -> None:
        
        if bool_train and bool_transform:
            # train
            if dataset_name == 'tinyimagenet':
                download_tinyimagenet()
                self.ds = ConcatDataset([
                    datasets.ImageFolder('/datasets/tiny-imagenet-200/train', transform=cls_datasets[dataset_name]['transforms']['train']), 
                    datasets.ImageFolder('/datasets/tiny-imagenet-200/val', transform=cls_datasets[dataset_name]['transforms']['train'])
                ])
            elif dataset_name == 'svhn':
                self.ds: Dataset = cls_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', split='train', download=True,
                    transform=cls_datasets[dataset_name]['transforms']['train']
                )
            else:
                self.ds: Dataset = cls_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', train=bool_train, download=True,
                    transform=cls_datasets[dataset_name]['transforms']['train']
                )    
            
        else:
            # validation or test
            if dataset_name == 'tinyimagenet':
                download_tinyimagenet()
                if bool_train:
                    self.ds = ConcatDataset([
                        datasets.ImageFolder('/datasets/tiny-imagenet-200/train', transform=cls_datasets[dataset_name]['transforms']['test']), 
                        datasets.ImageFolder('/datasets/tiny-imagenet-200/val', transform=cls_datasets[dataset_name]['transforms']['test'])
                    ])
                else:
                    self.ds: Dataset = datasets.ImageFolder('/datasets/tiny-imagenet-200/test', transform=cls_datasets[dataset_name]['transforms']['test'])
            elif dataset_name == 'svhn':
                self.ds: Dataset = cls_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', split='train' if bool_train else 'test', download=True,
                    transform=cls_datasets[dataset_name]['transforms']['test']
                )
            else:
                self.ds: Dataset = cls_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', train=bool_train, download=True,
                    transform=cls_datasets[dataset_name]['transforms']['test']
                )           
        
        
    def __len__(self) -> int:
        return len(self.ds) # type: ignore


    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, int]:
                
        image, label = self.ds[index]
        return index, image, label

    