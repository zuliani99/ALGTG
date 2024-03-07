
from torch.utils.data import Dataset, Subset

from torchvision import datasets
from torchvision import transforms
import torch

import numpy as np


becnhmark_datasets = {
    'cifar10': {
        'id': 1,
        'method': datasets.CIFAR10,
        'n_classes': 10,
        'channels': 3,
        'transforms': {
            'train': transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(np.array([0.49139968, 0.48215841, 0.44653091]),
                                             np.array([0.24703223, 0.24348513, 0.26158784]))
                    ]),
            'test': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(np.array([0.49139968, 0.48215841, 0.44653091]),
                                             np.array([0.24703223, 0.24348513, 0.26158784]))
                    ])
        }
    },
    'cifar100': {
        'id': 2,
        'method': datasets.CIFAR100,
        'n_classes': 100,
        'channels': 3,
        'transforms': {
            'train': transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(np.array([0.50707516, 0.48654887, 0.44091784]),
                                             np.array([0.26733429, 0.25643846, 0.27615047]))
                    ]),
            'test': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(np.array([0.50707516, 0.48654887, 0.44091784]),
                                             np.array([0.26733429, 0.25643846, 0.27615047]))
                    ])
        }
    },
    'fmnist': {
        'id': 3,
        'n_classes': 10,
        'method': datasets.FashionMNIST,
        'channels': 1,
        'transforms': {
            'train': transforms.Compose([
                        transforms.Pad(2),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(0.21899983206954657, 0.3318113729999592)
                    ]),
            'test': transforms.Compose([
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Normalize(0.21899983206954657, 0.3318113729999592)
                    ])
        }
    }
}


class SubsetDataloaders():
    
    def __init__(self, dataset_name,  batch_size, val_rateo, init_lab_obs, al_iters):
        self.batch_size = batch_size
        
        self.transformed_trainset = DatasetChoice(dataset_name=dataset_name, bool_train=True, bool_transform=True, al_iters=al_iters)
        self.non_transformed_trainset = DatasetChoice(dataset_name=dataset_name, bool_train=True, bool_transform=False)
        
        self.test_ds = DatasetChoice(dataset_name=dataset_name, bool_train=False, bool_transform=False)

        self.n_classes = becnhmark_datasets[dataset_name]['n_classes']
        self.n_channels = becnhmark_datasets[dataset_name]['channels']
        self.dataset_id = becnhmark_datasets[dataset_name]['id']
    
        self.get_initial_subsets_dls(val_rateo, init_lab_obs)
    
    
    
    def get_initial_subsets_dls(self, val_rateo, init_lab_obs):

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
        self.labeled_indices = train_indices[:init_lab_obs].tolist()
        self.unlabeled_indices = train_indices[init_lab_obs:].tolist()

    


class DatasetChoice(Dataset):
    def __init__(self, dataset_name, bool_train, bool_transform = True, al_iters = None):
        
        self.bool_transform = bool_transform
        self.al_iters = al_iters

        if bool_transform:
            # train
            self.ds = becnhmark_datasets[dataset_name]['method'](f'./datasets/{dataset_name}',train=bool_train, download=True,
                transform=becnhmark_datasets[dataset_name]['transforms']['train']
            )    
            
        else:
            # validation or test
            self.ds = becnhmark_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', train=bool_train, download=True,
                transform=becnhmark_datasets[dataset_name]['transforms']['test']
            )           
        
        
    def __len__(self):
        return len(self.ds)


    def __getitem__(self, index):
        
        image, label = self.ds[index]    
        return index, image, label

    