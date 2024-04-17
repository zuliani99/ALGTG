
from torchvision import datasets
from torchvision.transforms import v2
import torch


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
                        v2.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                     std=[0.24703223, 0.24348513, 0.26158784])
                        
                        
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                     std=[0.24703223, 0.24348513, 0.26158784])
                    ])
        }
    },
    'cifar100': {
        'id': 2,
        'method': datasets.CIFAR100,
        'n_classes': 100,
        'channels': 3,
        'image_size': 32,
        'classes': ["apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle","bowl","boy","bridge",
                    "bus","butterfly","camel","can","castle","caterpillar","cattle","chair","chimpanzee","clock","cloud","cockroach",
                    "couch","crab","crocodile","cup","dinosaur","dolphin","elephant","flatfish","forest","fox","girl","hamster","house",
                    "kangaroo","keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
                    "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree","plain","plate",
                    "poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket","rose","sea","seal","shark","shrew","skunk",
                    "skyscraper","snail","snake","spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone",
                    "television","tiger","tractor","train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"],
        'transforms': {
            'train': v2.Compose([
                        v2.ToImage(),
                        v2.RandomCrop(32, padding=4),
                        v2.RandomHorizontalFlip(),
                        v2.RandomRotation(degrees=(0,15)),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=[0.50707516, 0.48654887, 0.44091784],
                                     std=[0.26733429, 0.25643846, 0.27615047])
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=[0.50707516, 0.48654887, 0.44091784],
                                     std=[0.26733429, 0.25643846, 0.27615047])
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
                        v2.Normalize(mean=[0.4376821 , 0.4437697 , 0.47280442],
                                     std=[0.19803012, 0.20101562, 0.19703614])
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=[0.4376821 , 0.4437697 , 0.47280442],
                                     std=[0.19803012, 0.20101562, 0.19703614])
                    ])
        }
    },
    'tinyimagenet': {
        'id': 4,
        'n_classes': 200,
        'channels': 3,
        'image_size': 64,
        'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', 
                    '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', 
                    '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', 
                    '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', 
                    '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
                    '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', 
                    '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', 
                    '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', 
                    '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199'],
        'transforms': {
            'train': v2.Compose([
                        v2.ToImage(),
                        v2.RandomCrop(64, padding=8),
                        v2.RandomHorizontalFlip(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=[0.48042979, 0.44819701, 0.39755623],
                                     std=[0.27643974, 0.26888656, 0.28166852])
                    ]),
            'test': v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=[0.48042979, 0.44819701, 0.39755623],
                                     std=[0.27643974, 0.26888656, 0.28166852])
                    ])
        }
    }
}



# later added to argparser

al_params = {
    'init_lab_obs': 1000,
    'al_iters': 10, 
    'unlab_sample_dim': 10000, 
    'n_top_k_obs': 1000,
}
    

cls_config = {
    'epochs': 200,
    'batch_size': 128,
    'results_dict': { 'train': {'train_accuracy': [], 'train_loss': [], 'train_loss_ce': [], 'train_pred_loss': []},
                     'test': {'test_accuracy': [], 'test_loss': [], 'test_loss_ce': [], 'test_pred_loss': []}},
    'ds_params': {
        'cifar10': {
            'optimizer': torch.optim.SGD,
            'optim_p': {
                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4
            }
        },
        'cifar100': {
            'optimizer': torch.optim.SGD,
            'optim_p': {
                'lr': 0.1, 
                'momentum': 0.9,
                'weight_decay': 5e-4   
            }
        },
        'svhn': {
            'optimizer': torch.optim.SGD,
            'optim_p': {
                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 5e-4
            }
        },
        'caltech256': {
            'optimizer': torch.optim.SGD,
            'optim_p': {
                'lr': 0.001,
                'momentum': 0.9,
                'weight_decay': 5e-4
            }
        },
        'tinyimagenet': {
            'optimizer': torch.optim.SGD,
            'optim_p': {
                'lr': 0.001,
                'momentum': 0.9,
                'weight_decay': 5e-4
            }
        }
    }
}












det_datasets = {
    'voc': {
        'id': 5,
        'n_classes': 21,
        'channels': 3,
        'image_size': 300,
        'dataset_path': 'datasets/voc/VOCdevkit', 
        
        'classes': [#'background',
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'
            ],
    }
}


voc_config = {

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
    'lr_steps': (80000, 100000, 120000),
    'epochs': 300,
    'batch_size': 32,
    'results_dict': { 'train': { 'train_loss': [] , 'train_loc_loss': [], 'train_conf_loss': [], 'train_pred_loss': []},
                     'test': {'test_map': []}
                    }
}