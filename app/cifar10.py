from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch


class CIFAR10(Dataset):
    def __init__(self, bool_train, new_dataset=None, transform=None):
        #self.cifar10 = new_dataset if bool_train == None else datasets.CIFAR10('./cifar10', train=bool_train, download=True, transform=transform)
        if bool_train == None:
            self.cifar10 = new_dataset
        else:
            self.cifar10 = datasets.CIFAR10('./cifar10', train=bool_train, download=True, transform=transform)

        
    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        '''if len(self.cifar10[index]) == 3:
            idx, image, label = self.cifar10[index]
            #print('--------------------------------------', index, idx)
            return idx, image, label
        else:'''
        image, label = self.cifar10[index]
        return index, image, label
    
    
    
def get_cifar10(batch_size):

    trainset = CIFAR10(bool_train=True, transform=transforms.ToTensor())

    testset = CIFAR10(bool_train=False, transform=transforms.ToTensor())
    test_dl = DataLoader(testset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    return trainset, test_dl, classes