
from torch.utils.data import Dataset, DataLoader, Sampler, DataLoader, Subset

from torchvision import datasets
from torchvision import transforms
import torch

import numpy as np



becnhmark_datasets = {
    'cifar10': {
        'method': datasets.CIFAR10,
        'n_classes': 10,
    },
    'cifar100': {
        'method': datasets.CIFAR100,
        'n_classes': 100,
        
    },
    'fmnist': {
        'n_classes': 10,
        'method': datasets.FashionMNIST,
    }
}


class SubsetDataloaders():
    
    def __init__(self, dataset_name,  batch_size, val_rateo, labeled_ratio, al_iters):
        self.batch_size = batch_size
        
        self.transformed_trainset = DatasetChoice(dataset_name=dataset_name, bool_train=True, bool_transform=True, al_iters=al_iters)
        self.non_transformed_trainset = DatasetChoice(dataset_name=dataset_name, bool_train=True, bool_transform=False)
        
        self.test_dl = DataLoader(
            DatasetChoice(dataset_name=dataset_name, bool_train=False, bool_transform=False), 
            self.batch_size, 
            shuffle=False, 
            pin_memory=True
        )

        self.n_classes = becnhmark_datasets[dataset_name]['n_classes']
    
        self.get_initial_subsets_dls(val_rateo, labeled_ratio)
    
    
    
    def get_initial_subsets_dls(self, val_rateo, labeled_ratio):

        train_size = len(self.transformed_trainset)
        
        # computing the indice for the train and validation sets
        val_size = int(train_size * val_rateo)
        new_train_size = train_size - val_size
        
        # random shuffle of the train indices
        shuffled_indices = torch.randperm(train_size)
        
        # indices for the train and validation sets
        train_indices = shuffled_indices[:new_train_size]
        validation_indices = shuffled_indices[new_train_size:]
                
        # validation dataloader
        self.val_dl = DataLoader(
            Subset(self.non_transformed_trainset, validation_indices.tolist()),
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=True
        )
                
        # calculate the number of samples for each split
        labeled_size = int(labeled_ratio * new_train_size)

        # indices for the labeled and unlabeled sets      
        labeled_indices = train_indices[:labeled_size]
        unlabeled_indices = train_indices[labeled_size:]

        # subset for the labeled and unlabeled sets
        self.lab_train_subset = Subset(self.transformed_trainset, labeled_indices.tolist())
        self.unlab_train_subset = Subset(self.non_transformed_trainset, unlabeled_indices.tolist())
    


class DatasetChoice(Dataset):
    def __init__(self, dataset_name, bool_train, bool_transform = True, al_iters = None):
        
        self.bool_transform = bool_transform
        self.al_iters = al_iters
        self.get_train_mean_std(dataset_name)

        if bool_transform:
            # train
            self.ds = becnhmark_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', train=bool_train, download=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.train_mean, self.train_std)
            ]))
        else:
            # validation or test
            self.ds = becnhmark_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', train=bool_train, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.train_mean, self.train_std)
            ]))
            
            
    def get_train_mean_std(self, dataset_name):

        train_data = becnhmark_datasets[dataset_name]['method'](f'./datasets/{dataset_name}', train=True, download=True)
        
        x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
        
        self.train_mean = np.mean(x, axis=(0, 1)) / 255
        self.train_std = np.std(x, axis=(0, 1)) / 255
        
            
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        
        image, label = self.ds[index]    
        return index, image, label



class UniqueShuffle(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = dataset.indices
        self.shuffle_indices()

    def shuffle_indices(self):
        self.indices = list(torch.randperm(len(self.indices)))
        
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.dataset)
    
    

