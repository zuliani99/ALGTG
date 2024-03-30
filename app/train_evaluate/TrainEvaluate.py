
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
from torchvision.utils import save_image

from train_evaluate.Train_DDP import train, train_ddp
from ResNet18 import BasicBlock, ResNet_Weird
from Datasets import DatasetChoice, SubsetDataloaders
from utils import save_train_val_curves, write_csv, plot_new_labeled_tsne, create_method_res_dir, create_class_dir

import copy
from typing import List, Dict, Any

import logging
logger = logging.getLogger(__name__)


    
    
class TrainEvaluate(object):

    def __init__(self, training_params: Dict[str, Any], LL: bool) -> None:

        self.LL = LL        
        sdl: SubsetDataloaders = training_params['DatasetChoice']
        
        self.n_classes = sdl.n_classes
        self.n_channels = sdl.n_channels
        self.dataset_id = sdl.dataset_id
        self.image_size = sdl.image_size
        self.classes = sdl.classes
        
        self.test_ds: DatasetChoice = sdl.test_ds
        self.val_ds: Subset = sdl.val_ds
        
        self.transformed_trainset: DatasetChoice = sdl.transformed_trainset 
        self.non_transformed_trainset: DatasetChoice = sdl.non_transformed_trainset 
        
        self.device: torch.device = training_params['device']
        self.batch_size: int = training_params['batch_size']
       
        self.patience: int = training_params['patience']
        self.score_fn: function = training_params['score_fn']
        self.timestamp: str = training_params['timestamp']
        self.dataset_name: str = training_params['dataset_name']
        self.samp_iter: int = training_params['samp_iter']

        self.best_check_filename = f'app/checkpoints/{self.dataset_name}'
                
        self.labeled_indices = copy.deepcopy(sdl.labeled_indices)
        self.unlabeled_indices = copy.deepcopy(sdl.unlabeled_indices)
        
        
        self.lab_train_dl = DataLoader(
            Subset(self.transformed_trainset, self.labeled_indices),
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
        
        self.model = ResNet_Weird(BasicBlock, [2, 2, 2, 2], image_size=self.image_size, num_classes=self.n_classes, n_channels=self.n_channels).to(self.device) # type: ignore
    
        self.world_size = torch.cuda.device_count()
        
        
        self.path = f'results/{self.timestamp}/{self.dataset_name}/{self.samp_iter}/{self.method_name}'
        create_method_res_dir(self.path)
        
        # save initial labeled images
        self.save_labeled_images(self.labeled_indices)
        
    
    
    def save_labeled_images(self, new_labeled_idxs: List[int]) -> None:
        logger.info(' => Saving the new labeled images for further visual analysis...')
        create_class_dir(self.path, self.iter, self.classes)
        for idx, img, lab in Subset(self.non_transformed_trainset, new_labeled_idxs):
            save_image(
                img, f'{self.path}/new_labeled_images/{self.iter}/{self.classes[lab]}/{idx}.png'
            )
        logger.info(' DONE\n')
        
        
        
    def get_embeddings(self, dataloader: DataLoader, dict_to_modify: Dict[str, Any]) -> None:
        
        if dist.is_available():
            if self.world_size > 1: device = 'cuda:0'
            else: device = 'cuda' 
        else: device = 'cpu'

        checkpoint: Dict = torch.load(f'{self.best_check_filename}/best_{self.method_name}_{device}.pth.tar', map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        self.model.eval()

        # again no gradients needed
        with torch.inference_mode():
            for idxs, images, labels in dataloader:
                
                if 'embedds' in dict_to_modify:
                    _, embed, _, _ = self.model(images.to(self.device))
                    dict_to_modify['embedds'] = torch.cat((dict_to_modify['embedds'], embed.squeeze()), dim=0)
                    
                if 'probs' in dict_to_modify:
                    outs, _, _, _ = self.model(images.to(self.device))
                    dict_to_modify['probs'] = torch.cat((dict_to_modify['probs'], outs.squeeze().cpu()), dim=0)
                    
                if 'labels' in dict_to_modify:
                    dict_to_modify['labels'] = torch.cat((dict_to_modify['labels'], labels), dim=0)
                if 'idxs' in dict_to_modify:
                    dict_to_modify['idxs'] = torch.cat((dict_to_modify['idxs'], idxs), dim=0)

        
        
        
    def save_tsne(self, samp_unlab_subset: Subset, idxs_new_labels: List[int], \
                  d_labels: Dict[str, int], al_iter: int, gtg_result_prediction = None) -> None:
        # plot the tsne graph for each iteration
        
        logger.info(' => Saving the TSNE embeddings plot with labeled, unlabeled and new labeled observations')
        
        unlab_train_dl = DataLoader(
            samp_unlab_subset, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
        
        # recompute the embedding to plot the tsne
        lab_embedds_dict = {
            'embedds': torch.empty((0, self.model.linear.in_features), dtype=torch.float32, device=self.device),
            'labels': torch.empty(0, dtype=torch.int8)
        }
        unlab_embedds_dict = {
            'embedds': torch.empty((0, self.model.linear.in_features), dtype=torch.float32, device=self.device),
            'labels': torch.empty(0, dtype=torch.int8)
        }
            
        logger.info(' Getting the embeddings...')
        self.get_embeddings(self.lab_train_dl, lab_embedds_dict)
        self.get_embeddings(unlab_train_dl, unlab_embedds_dict)
        
        logger.info(' Plotting...')
        plot_new_labeled_tsne(
            lab_embedds_dict, unlab_embedds_dict,
            al_iter, self.method_name,
            self.dataset_name, idxs_new_labels, self.classes, 
            self.timestamp, self.samp_iter, d_labels,
            gtg_result_prediction
        )
        
        logger.info(' DONE\n')
        
                


    #def update_sets(self, overall_topk: List[int], idx_samp_unlab_obs: int) -> None:
    def update_sets(self, overall_topk: List[int]) -> None:
        
        # save the new labeled images to further visual analysis
        self.save_labeled_images(overall_topk)
        
        # Update the labeeld and unlabeled training set
        logger.info(' => Modifing the Subsets and Dataloader')
        # extend with the overall_topk
        self.labeled_indices.extend(overall_topk)
        
        # remove new labeled observations
        for idx_to_remove in overall_topk: self.unlabeled_indices.remove(idx_to_remove)
        
        # sanity check
        if len(list(set(self.unlabeled_indices) & set(self.labeled_indices))) == 0:
            logger.info(' Intersection between indices is EMPTY')
        else: 
            logger.exception('NON EMPTY INDICES INTERSECTION')
            raise Exception('NON EMPTY INDICES INTERSECTION')

        # generate the new labeled DataLoader
        self.lab_train_dl = DataLoader(
            Subset(self.transformed_trainset, self.labeled_indices),
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
        
        logger.info(' DONE\n')
        
                
        


    def train_evaluate_save(self, epochs: int, lab_obs: int, iter: int, results: Dict[str, List[float]]) -> None:
                
        params = {
            'num_classes': self.n_classes, 'n_channels': self.n_channels, 'image_size': self.image_size,
            'LL': self.LL, 'patience': self.patience, 'dataset_name': self.dataset_name, 'method_name': self.method_name,
            # the training is done only on the labeled training set
            'train_ds': Subset(self.transformed_trainset, self.labeled_indices), 'val_ds': self.val_ds, 'test_ds': self.test_ds,
            'batch_size': self.batch_size, 'score_fn': self.score_fn, 'main_device': self.device
        }
        
        
        
        # Pipe for the itra-process communication of the results
        parent_conn, child_conn = mp.Pipe()
        
        # if we are using multiple gpus
        if self.world_size > 1:
            logger.info(' => RUNNING DISTRIBUTED TRAINING')
            
            # spawn the process
            mp.spawn(train_ddp, args=(self.world_size, params, epochs, child_conn, ), nprocs=self.world_size, join=True) # type: ignore
            # obtain the results
            while parent_conn.poll(): train_recv, test_recv = parent_conn.recv()
                
        else:
            logger.info(' => RUNNING TRAINING')
            
            # add the already created labeeld train dataloader
            train_recv, test_recv = train(params, epochs)

        
        logger.info(' DONE\n')
        
        train_results = {
            'model_name': self.method_name, 
            'results': {
                    'train_loss': train_recv[0], 'train_loss_ce': train_recv[1],
                    'train_loss_weird': train_recv[2], 'train_accuracy': train_recv[3], 
                    'val_loss': train_recv[4], 'val_loss_ce': train_recv[5],
                    'val_loss_weird': train_recv[6], 'val_accuracy': train_recv[7]
                }
        }
             
        test_accuracy, test_loss, test_loss_ce, test_loss_weird = test_recv
        
        logger.info('TESTING RESULTS -> test_accuracy: {:.6f}, test_loss: {:.6f}, test_loss_ce: {:.6f} , test_loss_weird: {:.6f}\n\n'.format(
                test_accuracy, test_loss, test_loss_ce, test_loss_weird ))
             
        results['test_accuracy'].append(test_accuracy)
        results['test_loss'].append(test_loss)
        results['test_loss_ce'].append(test_loss_ce)
        results['test_loss_weird'].append(test_loss_weird)

        write_csv(
            ts_dir = self.timestamp,
            dataset_name = self.dataset_name,
            head = ['method', 'iter', 'lab_obs', 'test_accuracy', 'test_loss', 'test_loss_ce', 'test_loss_weird'],
            values = [self.method_name, self.samp_iter, lab_obs, test_accuracy, test_loss, test_loss_ce, test_loss_weird]
        )
        
        save_train_val_curves(train_results, self.timestamp, self.dataset_name, iter, self.samp_iter, self.LL)

        
        
