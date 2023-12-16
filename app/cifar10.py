from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch


class CIFAR10(Dataset):
    def __init__(self, bool_train, new_dataset=None, transform=None):
        self.cifar10 = new_dataset if bool_train == None else datasets.CIFAR10('./cifar10', train=bool_train, download=True, transform=transform)

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        image, label = self.cifar10[index]
        return image, label
    
    
def get_cifar10(batch_size):
    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = CIFAR10(bool_train=True, transform=transform_cifar10)

    testset = CIFAR10(bool_train=False, transform=transform_cifar10)
    test_dl = DataLoader(testset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    return trainset, test_dl, classes