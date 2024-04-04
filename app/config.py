
from torchvision import datasets
from torchvision.transforms import v2
import torch

from utils import accuracy_score

cls_datasets = {
    'cifar10': {
        'id': 1,
        'method': datasets.CIFAR10,
        'n_classes': 10,
        'channels': 3,
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'image_size': 32, 
        'transforms': {
            'train': v2.Compose([
                        v2.ToImage(),
                        v2.RandomCrop(32, padding=4),
                        v2.RandomHorizontalFlip(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize((0.49139968, 0.48215841, 0.44653091),
                                     (0.24703223, 0.24348513, 0.26158784))
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize((0.49139968, 0.48215841, 0.44653091),
                                     (0.24703223, 0.24348513, 0.26158784))
                    ])
        }
    },
    'cifar100': {
        'id': 2,
        'method': datasets.CIFAR100,
        'n_classes': 100,
        'channels': 3,
        'image_size': 32,
        'classes' : [
            'beaver', 'dolphin', 'otter', 'seal', 'whale',
            'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
            'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
            'bottles', 'bowls', 'cans', 'cups', 'plates',
            'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
            'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
            'bed', 'chair', 'couch', 'table', 'wardrobe',
            'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
            'bear', 'leopard', 'lion', 'tiger', 'wolf',
            'bridge', 'castle', 'house', 'road', 'skyscraper',
            'cloud', 'forest', 'mountain', 'plain', 'sea',
            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
            'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
            'crab', 'lobster', 'snail', 'spider', 'worm',
            'baby', 'boy', 'girl', 'man', 'woman',
            'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
            'maple', 'oak', 'palm', 'pine', 'willow',
            'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
            'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
        ],
        'transforms': {
            'train': v2.Compose([
                        v2.ToImage(),
                        v2.RandomCrop(32, padding=4),
                        v2.RandomHorizontalFlip(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize((0.50707516, 0.48654887, 0.44091784),
                                     (0.26733429, 0.25643846, 0.27615047))
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize((0.50707516, 0.48654887, 0.44091784),
                                     (0.26733429, 0.25643846, 0.27615047))
                    ])
        }
    },
    'svhn': {
        'id': 3,
        'n_classes': 10,
        'method': datasets.SVHN,
        'channels': 3,
        'image_size': 32, 
        'classes': ['0','1','2','3','4','5','6','7','8','9'],
        'transforms': {
            'train': v2.Compose([
                        v2.ToImage(),
                        v2.RandomCrop(32, padding=4),
                        v2.RandomHorizontalFlip(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize((0.4376821 , 0.4437697 , 0.47280442),
                                     (0.19803012, 0.20101562, 0.19703614))
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize((0.4376821 , 0.4437697 , 0.47280442),
                                     (0.19803012, 0.20101562, 0.19703614))
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
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize((0.48042979, 0.44819701, 0.39755623),
                                     (0.27643974, 0.26888656, 0.28166852))
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize((0.48042979, 0.44819701, 0.39755623),
                                     (0.27643974, 0.26888656, 0.28166852))
                    ])
        }
    }
}



# later added to argparser

al_params = dict(
    al_iters = 10, 
    unlab_sample_dim = 1000, 
    n_top_k_obs = 10000,
)
    

cls_config = dict(
    epochs = 200,
    batch_size = 128,
    results_dict = { 'train': {'train_accuracy': [], 'train_loss': [] , 'train_loss_ce': [], 'train_loss_weird': []},
                     'test': {'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}}
)

det_datasets = {
    'voc': {
        'id': 5,
        'n_classes': 21,
        'channels': 3,
        'image_size': 300,
        'dataset_path': '/content/VOCdevkit', 
        'classes': [
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'
            ],
    }
}


voc_config = {
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}

det_config = {
    'batch_size': 32,
    'results_dict': { 'train': { 'train_loss': [] , 'train_loc_loss': [], 'train_conf_loss': [], 'train_loss_weird': []},
                     'test': {'test_map': [], 'test_loss': [] , 'test_loss_target': [], 'test_loss_weird': []}
                    }
}