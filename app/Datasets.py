
from torch.utils.data import Dataset, Subset, ConcatDataset

from torchvision import datasets
from torchvision.transforms import v2
import torch

import numpy as np

from typing import List, Tuple

from utils import download_tinyimagenet



becnhmark_datasets = {
    'cifar10': {
        'id': 1,
        'method': datasets.CIFAR10,
        'n_classes': 10,
        'channels': 3,
        'image_size': 32, 
        'transforms': {
            'train': v2.Compose([
                        v2.ToImage(),
                        v2.RandomCrop(32, padding=4),
                        v2.RandomHorizontalFlip(),
                        v2.RandomRotation(15),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(np.array([0.49139968, 0.48215841, 0.44653091]),
                                     np.array([0.24703223, 0.24348513, 0.26158784]))
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(np.array([0.49139968, 0.48215841, 0.44653091]),
                                     np.array([0.24703223, 0.24348513, 0.26158784]))
                    ])
        }
    },
    'cifar100': {
        'id': 2,
        'method': datasets.CIFAR100,
        'n_classes': 100,
        'channels': 3,
        'image_size': 32, 
        'transforms': {
            'train': v2.Compose([
                        v2.ToImage(),
                        v2.RandomCrop(32, padding=4),
                        v2.RandomHorizontalFlip(),
                        v2.RandomRotation(15),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(np.array([0.50707516, 0.48654887, 0.44091784]),
                                     np.array([0.26733429, 0.25643846, 0.27615047]))
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(np.array([0.50707516, 0.48654887, 0.44091784]),
                                     np.array([0.26733429, 0.25643846, 0.27615047]))
                    ])
        }
    },
    'svhn': {
        'id': 3,
        'n_classes': 10,
        'method': datasets.SVHN,
        'channels': 3,
        'image_size': 32, 
        'transforms': {
            'train': v2.Compose([
                        v2.ToImage(),
                        v2.RandomCrop(32, padding=4),
                        v2.RandomHorizontalFlip(),
                        v2.RandomRotation(15),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(np.array([0.4376821 , 0.4437697 , 0.47280442]),
                                     np.array([0.19803012, 0.20101562, 0.19703614]))
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(np.array([0.4376821 , 0.4437697 , 0.47280442]),
                                     np.array([0.19803012, 0.20101562, 0.19703614]))
                    ])
        }
    },
    'tinyimagenet': {
        'id': 4,
        'n_classes': 200,
        'channels': 3,
        'image_size': 64,
        'transforms': {
            'train': v2.Compose([
                        v2.ToImage(),
                        v2.RandomCrop(64, padding=4),
                        v2.RandomHorizontalFlip(),
                        v2.RandomRotation(15),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(np.array([0.48042979, 0.44819701, 0.39755623]),
                                     np.array([0.27643974, 0.26888656, 0.28166852]))
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(np.array([0.48042979, 0.44819701, 0.39755623]),
                                     np.array([0.27643974, 0.26888656, 0.28166852]))
                    ])
        }
    }
}


class SubsetDataloaders():
    
    def __init__(self, dataset_name: str, batch_size: int, val_rateo: int, init_lab_obs: int) -> None:
        self.batch_size = batch_size
        
        self.transformed_trainset = DatasetChoice(dataset_name=dataset_name, bool_train=True, bool_transform=True)
        self.non_transformed_trainset = DatasetChoice(dataset_name=dataset_name, bool_train=True, bool_transform=False)
        
        self.test_ds = DatasetChoice(dataset_name=dataset_name, bool_train=False, bool_transform=False)

        self.n_classes: int = becnhmark_datasets[dataset_name]['n_classes']
        self.n_channels: str = becnhmark_datasets[dataset_name]['channels']
        self.dataset_id: int = becnhmark_datasets[dataset_name]['id']
        self.image_size: int = becnhmark_datasets[dataset_name]['image_size']
    
        self.get_initial_subsets_dls(val_rateo, init_lab_obs)
    
    
    
    def get_initial_subsets_dls(self, val_rateo: int, init_lab_obs: int) -> None:

        train_size = len(self.transformed_trainset)

        # computing the size for the train and validation sets
        val_size = int(train_size * val_rateo)
        new_train_size = train_size - val_size
        
        # random shuffle of the train indices
        shuffled_indices = torch.randperm(train_size)
        # each time should be a new shuffle, thus a new train-validation, labeled-unlabeled split
            
        ##############################
        print(shuffled_indices[-5:])
        ##############################
            
        # indices for the train and validation sets
        train_indices = shuffled_indices[:new_train_size]
        validation_indices = shuffled_indices[new_train_size:]
                    
        # validation dataset
        self.val_ds = Subset(self.non_transformed_trainset, validation_indices.tolist())

        # indices for the labeled and unlabeled sets      
        self.labeled_indices: List[int] = train_indices[:init_lab_obs].tolist()
        self.unlabeled_indices: List[int] = train_indices[init_lab_obs:].tolist()

    


class DatasetChoice(Dataset):
    def __init__(self, dataset_name: str, bool_train: bool, bool_transform = True) -> None:
        
        self.bool_transform = bool_transform

        if bool_transform:
            # train
            if dataset_name == 'tinyimagenet':
                download_tinyimagenet()
                self.ds = ConcatDataset([
                    datasets.ImageFolder('/datasets/tiny-imagenet-200/train', transform=becnhmark_datasets[dataset_name]['transforms']['train']), 
                    datasets.ImageFolder('/datasets/tiny-imagenet-200/val', transform=becnhmark_datasets[dataset_name]['transforms']['train'])
                ])
            elif dataset_name == 'svhn':
                self.ds: Dataset = becnhmark_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', split='train' if bool_train else 'test', download=True,
                    transform=becnhmark_datasets[dataset_name]['transforms']['train']
                )
            else:
                self.ds: Dataset = becnhmark_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', train=bool_train, download=True,
                    transform=becnhmark_datasets[dataset_name]['transforms']['train']
                )    
            
        else:
            # validation or test
            if dataset_name == 'tinyimagenet':
                download_tinyimagenet()
                if bool_train:
                    self.ds = ConcatDataset([
                        datasets.ImageFolder('/datasets/tiny-imagenet-200/train', transform=becnhmark_datasets[dataset_name]['transforms']['test']), 
                        datasets.ImageFolder('/datasets/tiny-imagenet-200/val', transform=becnhmark_datasets[dataset_name]['transforms']['test'])
                    ])
                else:
                    self.ds: Dataset = datasets.ImageFolder('/datasets/tiny-imagenet-200/test', transform=becnhmark_datasets[dataset_name]['transforms']['test'])
            elif dataset_name == 'svhn':
                self.ds: Dataset = becnhmark_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', split='train' if bool_train else 'test', download=True,
                    transform=becnhmark_datasets[dataset_name]['transforms']['test']
                )
            else:
                self.ds: Dataset = becnhmark_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', train=bool_train, download=True,
                    transform=becnhmark_datasets[dataset_name]['transforms']['test']
                )           
        
        
    def __len__(self) -> int:
        return len(self.ds)


    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, int]:
                
        image, label = self.ds[index]
        return index, image, label

    