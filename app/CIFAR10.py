
from torch.utils.data import Dataset, DataLoader, Sampler, DataLoader, Subset, random_split

from torchvision import datasets
from torchvision import transforms
import torch

import numpy as np


class Cifar10SubsetDataloaders():
    
    def __init__(self, batch_size, val_rateo, labeled_ratio, flag_mean_std_train = False):
        self.batch_size = batch_size
        
        self.original_trainset = CIFAR10(bool_train=True)
        # heare there are the indices of the labeled observation to transform
        
        # flag to decide the normalization procedure
        self.flag_mean_std_train = flag_mean_std_train
        
        self.test_dl = DataLoader(CIFAR10(bool_train=False), self.batch_size, shuffle=False, num_workers=1, pin_memory=True)

        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
        self.get_initial_dataloaders(val_rateo, labeled_ratio)
        if self.flag_mean_std_train: self.original_trainset.get_std_mean_train()
    
    
    
    def get_initial_dataloaders(self, val_rateo, labeled_ratio):

        train_size = len(self.original_trainset) #50000
        
        val_size = int(train_size * val_rateo)
        new_train_size = train_size - val_size

        train_data, val_data = random_split(self.original_trainset, [int(new_train_size), int(val_size)])

        # validation dataloader
        self.val_dl = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=1, pin_memory=True)
                
        # Calculate the number of samples for each split
        labeled_size = int(labeled_ratio * new_train_size)
        unlabeled_size = new_train_size - labeled_size

        # Get the dataset split
        labeled_set, unlabeled_set = random_split(train_data, [labeled_size, unlabeled_size])

        self.lab_train_subset = Subset(self.original_trainset, [train_data.indices[id] for id in labeled_set.indices])
        self.unlab_train_subset = Subset(self.original_trainset, [train_data.indices[id] for id in unlabeled_set.indices])
        
        # Obtain the splitted dataloader
        self.lab_train_dl = DataLoader(labeled_set, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    
        self.original_trainset.lab_train_idxs = self.lab_train_subset.indices

    


class CIFAR10(Dataset):
    def __init__(self, bool_train):
        self.cifar10 = datasets.CIFAR10('./cifar10', train=bool_train, download=True)
        
        self.transform_labeled = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.transform = transforms.Compose( [transforms.ToTensor()] )
        
        self.lab_train_idxs = None
        self.flag_normalization = False
        
        self.bool_train = bool_train
        
        
    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        image, label = self.cifar10[index]
                
        if not self.flag_normalization and self.bool_train == True and index in self.lab_train_idxs:
            image = self.transform_labeled(image)
        else:
            image = self.transform(image)
            
        return index, image, label
    
    
    def get_std_mean_train(self):
        self.flag_normalization = True
        x = np.concatenate([np.asarray(self.cifar10[i][0]) for i in range(len(self.cifar10))])
        self.flag_normalization = False
        
        
        # calculate the mean and std along the (0, 1) axes
        self.train_mean = np.mean(x, axis=(0, 1)) / 255
        self.train_std = np.std(x, axis=(0, 1)) / 255
        
    

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
    
    

