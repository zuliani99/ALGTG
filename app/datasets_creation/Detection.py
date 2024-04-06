
import torch
from torchvision import datasets

from datasets_creation.detection_ds.voc.voc0712 import VOCAnnotationTransform, VOCDetection
from datasets_creation.detection_ds.voc.voc_utils import BaseTransform, SSDAugmentation

from config import det_datasets, voc_config

from typing import List, Tuple

import logging
logger = logging.getLogger(__name__)



def get_detection_dataset(ds_name: str, dataset_path: str) -> Tuple[VOCDetection, VOCDetection, VOCDetection]:
    #if ds_name == 'voc':
    logger.info(' => Downloading PASCAL VOC 2007 (train + test) and 2012 (train)')
    datasets = get_voc_data(dataset_path)
    logger.info(' DONE\n')
    return datasets
        

def get_voc_data(dataset_path:str) -> Tuple[VOCDetection, VOCDetection, VOCDetection]:
    
    voc_means = (104, 117, 123)
    
    datasets.VOCDetection(root=dataset_path, year='2007', image_set='train', download=True)
    datasets.VOCDetection(root=dataset_path, year='2012', image_set='train', download=True)
    datasets.VOCDetection(root=dataset_path, year='2007', image_set='test', download=True)

    
    train = VOCDetection(root=f'{dataset_path}/VOCdevkit',
                                    transform=SSDAugmentation(voc_config['min_dim'], voc_means))
    unlabeled = VOCDetection(root=f'{dataset_path}/VOCdevkit',
                                        transform=BaseTransform(300, voc_means),
                                        target_transform=VOCAnnotationTransform())
    test = VOCDetection(root=f'{dataset_path}/VOCdevkit',
                                   image_sets=[('2007', 'test')],
                                   transform=BaseTransform(300, voc_means),
                                   target_transform=VOCAnnotationTransform())
    return train, unlabeled, test



def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    index, imgs, targets = [], [], []
    for sample in batch:
        index.append(sample[0])
        imgs.append(sample[1])
        targets.append(torch.FloatTensor(sample[2]))
    return index, torch.stack(imgs, 0), targets



class Det_Datasets():
    def __init__(self, dataset_name: str, init_lab_obs: int) -> None:

        self.transformed_trainset, self.non_transformed_trainset, self.test_ds = get_detection_dataset(dataset_name, 'datasets/voc')
        
        self.n_classes: int = det_datasets[dataset_name]['n_classes']
        self.dataset_id: int = det_datasets[dataset_name]['id']
        self.classes: List[str] = det_datasets[dataset_name]['classes']
        self.n_channels: int = det_datasets[dataset_name]['channels']
        self.dataset_path: str = det_datasets[dataset_name]['dataset_path']
        
        self.get_initial_subsets(init_lab_obs)
    

    
    def get_initial_subsets(self, init_lab_obs: int) -> None:

        train_size = len(self.transformed_trainset)
        
        # random shuffle of the train indices
        shuffled_indices = torch.randperm(train_size)
        # each time should be a new shuffle, thus a new train-validation, labeled-unlabeled split

        logger.info(f' Last 5 shuffled train observations: {shuffled_indices[-5:]}')

        # indices for the labeled and unlabeled sets
        self.labeled_indices: List[int] = shuffled_indices[:init_lab_obs].tolist()
        self.unlabeled_indices: List[int] = shuffled_indices[init_lab_obs:].tolist()
