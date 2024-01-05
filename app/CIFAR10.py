
from torch.utils.data import Dataset, DataLoader, Sampler, DataLoader, Subset, random_split

from torchvision import datasets
from torchvision.transforms import v2
import torch


class CIFAR10(Dataset):
    def __init__(self, bool_train):
        self.cifar10 = datasets.CIFAR10('./cifar10', train=bool_train, download=True)
        
        self.transform_labeled = v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.lab_train_idxs = None
        
    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        image, label = self.cifar10[index]
        
        #print(self.lab_train_idxs)
        
        if self.lab_train_idxs is not None and index in self.lab_train_idxs:
            image = self.transform_labeled(image)
            print('okok', self.lab_train_idxs)
        else:
            #print('non okok', self.lab_train_idxs)
            image = self.transform(image)
            
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
    
    

class Cifar10SubsetDataloaders():
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.original_trainset = CIFAR10(bool_train=True)

        self.testset = CIFAR10(bool_train=False)
        
        self.test_dl = DataLoader(self.testset, self.batch_size, shuffle=False, num_workers=1, pin_memory=True)

        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    
    
    def get_initial_dataloaders(self, val_rateo, labeled_ratio):

        train_size = len(self.original_trainset) #50000

        val_size = int(train_size * val_rateo)
        train_size -= val_size

        train_data, val_data = random_split(self.original_trainset, [int(train_size), int(val_size)])

        # validation dataloader
        self.val_dl = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=1, pin_memory=True)

        train_data_size = len(train_data)

        # Calculate the number of samples for each split
        labeled_size = int(labeled_ratio * train_data_size)
        unlabeled_size = train_data_size - labeled_size

        # Get the dataset split
        labeled_set, unlabeled_set = random_split(train_data, [labeled_size, unlabeled_size])

        self.lab_train_subset = Subset(self.original_trainset, [train_data.indices[id] for id in labeled_set.indices])
        self.unlab_train_subset = Subset(self.original_trainset, [train_data.indices[id] for id in unlabeled_set.indices])
        
        # Obtain the splitted dataloader
        self.lab_train_dl = DataLoader(labeled_set, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True)

        self.original_trainset.lab_train_idxs = self.lab_train_subset.indices
    