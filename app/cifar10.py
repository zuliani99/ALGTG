
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, transforms
import torch


class CIFAR10(Dataset):
    def __init__(self, bool_train, transform_labeled = None, transform = transforms.Compose([transforms.ToTensor()])):
        self.cifar10 = datasets.CIFAR10('./cifar10', train=bool_train, download=True)
        
        self.transform_labeled = transform_labeled
        self.transform = transform
        self.lab_train_idxs = None
        
    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        image, label = self.cifar10[index]
        
        if self.lab_train_idxs is not None and index in self.lab_train_idxs:
            image = self.transform_labeled(image)
        else: 
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
    
    
def get_cifar10(batch_size):

    trainset = CIFAR10(bool_train=True, transform_labeled=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))

    testset = CIFAR10(bool_train=False)
    
    #test_dl = DataLoader(testset, batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_dl = DataLoader(testset, batch_size, shuffle=False, num_workers=1, pin_memory=True)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    return trainset, test_dl, classes