
from torch.utils.data import Dataset, DataLoader, Sampler, DataLoader, Subset

from torchvision import datasets
from torchvision import transforms
import torch

import numpy as np


class Cifar10SubsetDataloaders():
    
    def __init__(self, batch_size, val_rateo, labeled_ratio):
        self.batch_size = batch_size
        
        self.transformed_trainset = CIFAR10(bool_train=True)
        self.non_transformed_trainset = CIFAR10(bool_train=True, bool_transform=False)
        
        self.test_dl = DataLoader(CIFAR10(bool_train=False, bool_transform=False), self.batch_size, shuffle=False, pin_memory=True)

        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
        self.get_initial_subsets_dls(val_rateo, labeled_ratio)
    
    
    
    def get_initial_subsets_dls(self, val_rateo, labeled_ratio):

        train_size = len(self.transformed_trainset) #50000
        
        # computing the indice for the train and validation sets
        val_size = int(train_size * val_rateo)
        new_train_size = train_size - val_size
        
        # random shuffle of the trian indices
        shuffled_indices = np.arange(train_size)
        np.random.shuffle(shuffled_indices)
        
        # indices for the train and validation sets
        random_indices_train_val = np.random.choice(len(shuffled_indices), size=new_train_size, replace=False)
        train_indices = shuffled_indices[random_indices_train_val]
        validation_indices = np.delete(shuffled_indices, random_indices_train_val)
                
        # validation dataloader
        self.val_dl = DataLoader(Subset(self.non_transformed_trainset, validation_indices.tolist()), batch_size=self.batch_size, shuffle=False, pin_memory=True)
                
        # calculate the number of samples for each split
        labeled_size = int(labeled_ratio * new_train_size)

        # indices for the labeled and unlabeled sets
        random_indices_lab_unlab = np.random.choice(len(train_indices), size=labeled_size, replace=False)
        labeled_indices = train_indices[random_indices_lab_unlab]
        unlabeled_indices = np.delete(train_indices, random_indices_lab_unlab)

        # subset for the labeled and unlabeled sets
        self.lab_train_subset = Subset(self.transformed_trainset, labeled_indices.tolist())
        self.unlab_train_subset = Subset(self.non_transformed_trainset, unlabeled_indices.tolist())
    

class CIFAR10(Dataset):
    def __init__(self, bool_train, bool_transform = True):

        if bool_train and bool_transform:
            # train
            self.cifar10 = datasets.CIFAR10('./cifar10', train=bool_train, download=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                                    [0.24703225141799082, 0.24348516474564, 0.26158783926049628])
                ]))
        else:
            # validation or test
            self.cifar10 = datasets.CIFAR10('./cifar10', train=bool_train, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                                    [0.24703225141799082, 0.24348516474564, 0.26158783926049628])
                ]))
            
    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        image, label = self.cifar10[index]
            
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
    
    

