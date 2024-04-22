
from torch.utils.data import Dataset, Subset
from torchvision import datasets
import torch
import numpy as np

from typing import List, Tuple
import os
import shutil

from torchvision.transforms import v2

from utils import count_class_observation
from config import cls_datasets 

import logging
logger = logging.getLogger(__name__)



def download_tinyimagenet() -> None:
    if not os.path.exists('datasets/tiny-imagenet-200'):
        logger.info(' => Downloading Tiny-IMAGENET Dataset')
        os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        os.system('unzip tiny-imagenet-200.zip -d datasets')
        os.remove('tiny-imagenet-200.zip')
        logger.info(' DONE\n')
    else:
        logger.info('Tiny-IMAGENET Dataset already downloaded')
        
        
def create_val_img_folder(dataset_dir):
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


def preprocess_caltech256_dir(caltech256_path):
    n_images = 0
    for dir_name in os.listdir(caltech256_path):
        new_dir_name = dir_name.split('.')[-1]
        os.rename(os.path.join(caltech256_path, dir_name), os.path.join(caltech256_path, new_dir_name))
        for file_name in os.listdir(os.path.join(caltech256_path, new_dir_name)):
            n_images += 1
            new_file_name = file_name.split('_')[-1]
            os.rename(os.path.join(caltech256_path, new_dir_name, file_name),
                      os.path.join(caltech256_path, new_dir_name, new_file_name))
    return n_images


def download_caltech256():
    os.makedirs('./datasets/caltech256', exist_ok=True)
    logger.info(' => Downloading Caltech256 Dataset')
    # downlaod dataset
    os.system('wget -P ./datasets/caltech256 https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar')
    # extract and remove it
    os.system('tar -xopf ./datasets/caltech256/256_ObjectCategories.tar -C ./datasets/caltech256')
    os.remove('./datasets/caltech256/256_ObjectCategories.tar')
    # remove non-images
    os.remove(os.path.join('./datasets/caltech256/256_ObjectCategories', '198.spider/RENAME2'))
    shutil.rmtree(os.path.join('./datasets/caltech256/256_ObjectCategories', '056.dog/greg'))
    # we don't need the class with noise
    shutil.rmtree(os.path.join('./datasets/caltech256/256_ObjectCategories', '257.clutter'))
        
    return preprocess_caltech256_dir('./datasets/caltech256/256_ObjectCategories')


def create_train_test_dir(list_idxs: Tuple[List[int], List[int]], train_data: Dataset, split_path: str, \
                          split_types: Tuple[str, str], classes: List[str]) -> None:
    for idxs, split_type in zip(list_idxs, split_types):
        for i in idxs:
            pil_image, label = train_data[i]
            image_path = os.path.join(split_path, split_type, classes[label])
            os.makedirs(image_path, exist_ok=True)
            id = len(os.listdir(image_path)) + 1
            pil_image.save(os.path.join(split_path, split_type, classes[int(label)], f'{id}.png'), 'PNG')


def compute_mean_std(data: Dataset, idxs=None) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(' Computing the mean and std for the train set')
    images = []
    if idxs != None:
        for i in idxs:
            if len(np.asarray(data[i][0]).shape) == 3: images.append(np.asarray(data[i][0]))
            else: images.append(np.repeat(np.asarray(data[i][0])[:, :, np.newaxis], 3, axis=2))
    else: 
        for img, _ in data: images.append(np.asarray(img))
    images = np.concatenate(images)
    mean = np.mean(images, axis=(0, 1)) / 255 
    std = np.std(images, axis=(0, 1)) / 255
    return mean, std


def complete_caltech_transforms(mean_std: Tuple[np.ndarray, np.ndarray]) -> None:
    mean, std = mean_std
    cls_datasets['caltech256']['transforms']['train'].append(v2.Normalize(mean=mean.tolist(), std=std.tolist()))
    cls_datasets['caltech256']['transforms']['train'] = v2.Compose(cls_datasets['caltech256']['transforms']['train'])
                
    cls_datasets['caltech256']['transforms']['test'].append(v2.Normalize(mean=mean.tolist(), std=std.tolist()))
    cls_datasets['caltech256']['transforms']['test'] = v2.Compose(cls_datasets['caltech256']['transforms']['test'])


def init_caltech256() -> None:
    if os.path.exists('./datasets/caltech256/256_ObjectCategories'): 
        logger.info(' Caltech256 dataset already obtained and ready!')
        cls_datasets['caltech256']['classes'] = sorted(os.listdir('./datasets/caltech256/256_ObjectCategories/train'))
        complete_caltech_transforms(compute_mean_std(datasets.ImageFolder('./datasets/caltech256/256_ObjectCategories')))
        return 
    
    n_images = download_caltech256()
    classes = sorted(os.listdir('./datasets/caltech256/256_ObjectCategories'))
    cls_datasets['caltech256']['classes'] = classes    
        
    logger.info(' => Generating new train test split datasets...')
    
    rand_perm = torch.randperm(n_images)
    
    # 80% for the training set and the remaning 20% for the test set
    train_size = int(n_images * 0.8)
    train_idxs = rand_perm[:train_size].tolist()
    test_idxs = rand_perm[train_size:].tolist()
                
    train_data = datasets.ImageFolder('./datasets/caltech256/256_ObjectCategories', transform=v2.Compose([v2.Resize((64,64))]))
        
    split_path = './datasets/caltech256/256_ObjectCategories_new'
    os.mkdir(split_path)
    create_train_test_dir((train_idxs, test_idxs), train_data, split_path, ('train', 'test'), classes)
    
    logger.info(' => Computing mean and std for the train test split...')                
    complete_caltech_transforms(compute_mean_std(train_data, train_idxs))

    shutil.rmtree('./datasets/caltech256/256_ObjectCategories')
    os.rename(split_path, './datasets/caltech256/256_ObjectCategories')

    logger.info(' DONE\n')



class Cls_Datasets():
    
    def __init__(self, dataset_name: str, init_lab_obs: int) -> None:
        
        self.transformed_trainset = Cls_Dataset(dataset_name=dataset_name, bool_train=True, bool_transform=True)
        self.non_transformed_trainset = Cls_Dataset(dataset_name=dataset_name, bool_train=True, bool_transform=False)
        
        self.test_ds = Cls_Dataset(dataset_name=dataset_name, bool_train=False, bool_transform=False)

        self.n_classes: int = cls_datasets[dataset_name]['n_classes']
        self.n_channels: int = cls_datasets[dataset_name]['channels']
        self.dataset_id: int = cls_datasets[dataset_name]['id']
        self.image_size: int = cls_datasets[dataset_name]['image_size']
 
        if dataset_name == 'tinyimagenet':
            cls_datasets[dataset_name]['classes'] = os.listdir('./datasets/tiny-imagenet-200/train')
            
        self.classes: List[str] = cls_datasets[dataset_name]['classes']
        
        self.get_initial_subsets(init_lab_obs)
    
    
    
    def get_initial_subsets(self, init_lab_obs: int) -> None:

        train_size = len(self.transformed_trainset)
        
        # random shuffle of the train indices
        shuffled_indices = torch.randperm(train_size)
        # each time should be a new shuffle, thus a new train-validation, labeled-unlabeled split

        # indices for the labeled and unlabeled sets
        self.labeled_indices: List[int] = shuffled_indices[:init_lab_obs].tolist()
        self.unlabeled_indices: List[int] = shuffled_indices[init_lab_obs:].tolist()
        
        logger.info(f' Initial subset of labeled observations composed with: {count_class_observation(
            self.classes, Subset(self.transformed_trainset, self.labeled_indices)
        )}')




class Cls_Dataset(Dataset):
    def __init__(self, dataset_name: str, bool_train: bool, bool_transform: bool) -> None:
        
        if bool_train and bool_transform:
            # train
            if dataset_name == 'tinyimagenet':
                download_tinyimagenet()
                create_val_img_folder('./datasets/tiny-imagenet-200')
                self.ds: Dataset = datasets.ImageFolder('./datasets/tiny-imagenet-200/train',
                    transform=cls_datasets['tinyimagenet']['transforms']['train']
                )
            elif dataset_name == 'svhn':
                self.ds: Dataset = cls_datasets['svhn']['method']('./datasets/svhn', 
                    split='train', download=True,
                    transform=cls_datasets['svhn']['transforms']['train']
                )
            elif dataset_name == 'caltech256':
                init_caltech256()
                self.ds: Dataset = datasets.ImageFolder('./datasets/caltech256/256_ObjectCategories/train',
                    transform=cls_datasets['caltech256']['transforms']['train']
                )
            else:
                self.ds: Dataset = cls_datasets[dataset_name]['method'](f'./datasets/{dataset_name}',
                    train=bool_train, download=True,
                    transform=cls_datasets[dataset_name]['transforms']['train']
                )    
        else:
            # unlabeled or test dataset
            if dataset_name == 'tinyimagenet':
                self.ds: Dataset = datasets.ImageFolder(f'./datasets/tiny-imagenet-200/{'train' if bool_train else 'val'}', 
                    transform=cls_datasets['tinyimagenet']['transforms']['test']
                )
            elif dataset_name == 'svhn':
                self.ds: Dataset = cls_datasets['svhn']['method']('./datasets/svhn', split='train' 
                    if bool_train else 'test', download=True,
                    transform=cls_datasets['svhn']['transforms']['test']
                )
            elif dataset_name == 'caltech256':
                self.ds: Dataset = datasets.ImageFolder(f'./datasets/caltech256/256_ObjectCategories/{'train' if bool_train else 'test'}', 
                    transform=cls_datasets['caltech256']['transforms']['test']
                )            
            else:
                self.ds: Dataset = cls_datasets[dataset_name]['method'](f'./datasets/{dataset_name}',
                    train=bool_train, download=True,
                    transform=cls_datasets[dataset_name]['transforms']['test']
                )           
        
        
    def __len__(self) -> int:
        return len(self.ds) # type: ignore


    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, int]:
        image, label = self.ds[index]
        return index, image, label