
import wandb

from models.ResNet18 import ResNet_LL
from datasets_creation.Classification import Cls_Datasets
from datasets_creation.Detection import Det_Datasets
from models.ssd_pytorch.SSD import SSD_LL
from train_evaluate.Train_DDP import train, train_ddp
from utils import count_class_observation, print_cumulative_train_results, set_seeds,\
    create_class_dir, create_method_res_dir, plot_new_labeled_tsne, save_train_val_curves, write_csv

from torch.utils.data import Subset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
import torch
import numpy as np

from typing import List, Dict, Any
import copy

import logging
logger = logging.getLogger(__name__)


class ActiveLearner():
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any]) -> None:
        
        self.iter = 1
        
        self.al_p: Dict[str, Any] = al_p
        self.ct_p: Dict[str, Any] = ct_p
        self.t_p: Dict[str, Any] = t_p
        
        self.model: SSD_LL | ResNet_LL = self.ct_p['Model']
        self.device: torch.device = self.ct_p['device']
        
        self.dataset: Cls_Datasets | Det_Datasets = self.ct_p['Dataset']
        
        self.best_check_filename: str = f'app/checkpoints/{self.ct_p['dataset_name']}'
        
        self.labeled_indices: List[int] = copy.deepcopy(self.dataset.labeled_indices)
        self.unlabeled_indices: List[int] = copy.deepcopy(self.dataset.unlabeled_indices)

        self.train_results: Dict[int, Any] = {}
        
        self.lab_train_dl = DataLoader(
            Subset(self.dataset.transformed_trainset, self.labeled_indices),
            batch_size=self.t_p['batch_size'], shuffle=False, pin_memory=True
        )
        
        self.world_size: int = torch.cuda.device_count()
        
        self.path = f'results/{self.ct_p['timestamp']}/{self.ct_p['dataset_name']}/{self.ct_p['trial']}/{self.method_name}'
        create_method_res_dir(self.path)
        
        # save initial labeled images
        self.save_labeled_images(self.labeled_indices)
        
        
        
    
    def save_labeled_images(self, new_labeled_idxs: List[int]) -> None:
        logger.info(f' => Iteration {self.iter} - Saving the new labeled images for further visual analysis...')
        create_class_dir(self.path, self.iter, self.dataset.classes)
        for idx, img, gt in Subset(self.dataset.non_transformed_trainset, new_labeled_idxs):
            if self.ct_p['task'] != 'clf': 
                unique_labs = np.unique(np.array([labs[-1] for labs in gt]))
                for lab in unique_labs: 
                    save_image(img, f'{self.path}/new_labeled_images/{self.iter}/{self.dataset.classes[int(lab)]}/{idx}.png')
            else:
                save_image(img, f'{self.path}/new_labeled_images/{self.iter}/{self.dataset.classes[gt]}/{idx}.png')
        logger.info(' DONE\n')
        
        
    
    def get_samp_unlab_subset(self) -> Subset:
        if self.ct_p['task'] == 'clf':
            # set seed for reproducibility
            seed = self.dataset.dataset_id * (self.ct_p['trial'] * self.al_p['al_iters'] + (self.iter - 1))
            set_seeds(seed)
            
            rand_perm = torch.randperm(len(self.unlabeled_indices)).tolist()
            rand_perm_unlabeled = [self.unlabeled_indices[idx] for idx in rand_perm[:self.al_p['unlab_sample_dim']]]
            
            logger.info(f' SEED: {seed} - Last 10 permuted indices are: {rand_perm[-10:]}')
            unlab_perm_subset = Subset(self.dataset.non_transformed_trainset, rand_perm_unlabeled)
            logger.info(f' SEED: {seed} - With dataset indices: {unlab_perm_subset.indices[-10:]}')
            
            #reset the original seed
            set_seeds()
        
            return unlab_perm_subset
        else:
            return Subset(self.dataset.non_transformed_trainset, self.unlabeled_indices)
    
    
    # CHANGE THIS FUNCTION ONCE WE ARE IN DECTECTION TASK
    ##################################################################################################################
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
                    _, embed = self.model(images.to(self.device))
                    dict_to_modify['embedds'] = torch.cat((dict_to_modify['embedds'], embed.squeeze()), dim=0)
                    
                if 'probs' in dict_to_modify:
                    outs, _  = self.model(images.to(self.device))
                    dict_to_modify['probs'] = torch.cat((dict_to_modify['probs'], outs.squeeze().cpu()), dim=0)
                    
                if 'pred_loss' in dict_to_modify:
                    _, pred_loss = self.model(images.to(self.device))
                    dict_to_modify['pred_loss'] = torch.cat((dict_to_modify['pred_loss'], pred_loss.squeeze().cpu()), dim=0)
                    
                if 'labels' in dict_to_modify:
                    dict_to_modify['labels'] = torch.cat((dict_to_modify['labels'], labels), dim=0)
                if 'idxs' in dict_to_modify:
                    dict_to_modify['idxs'] = torch.cat((dict_to_modify['idxs'], idxs), dim=0)
    ##################################################################################################################

    
    
    
    
    def save_tsne(self, samp_unlab_subset: Subset, idxs_new_labels: List[int], \
                  d_labels: Dict[str, int], al_iter: int, gtg_result_prediction = None) -> None:
        # plot the tsne graph for each iteration
        
        logger.info(' => Saving the TSNE embeddings plot with labeled, unlabeled and new labeled observations')
        
        unlab_train_dl = DataLoader(
            samp_unlab_subset, batch_size=self.t_p['batch_size'], shuffle=False, pin_memory=True
        )
        
        # recompute the embedding to plot the tsne
        # CHANGE THIS FUNCTION ONCE WE ARE IN DECTECTION TASK
        ##################################################################################################################
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
        ##################################################################################################################
        
        
        
        
        logger.info(' Plotting...')
        plot_new_labeled_tsne(
            lab_embedds_dict, unlab_embedds_dict,
            al_iter, self.method_name,
            self.ct_p['dataset_name'], idxs_new_labels, self.dataset.classes, 
            self.ct_p['timestamp'], self.ct_p['trial'], d_labels,
            gtg_result_prediction
        )
        
        logger.info(' DONE\n')
        
        del lab_embedds_dict
        del unlab_embedds_dict
        torch.cuda.empty_cache()
        
        
        
    def train_evaluate_save(self, lab_obs: int, iter: int, results_format: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        
        test_res_keys = list(results_format['test'].keys())
        train_res_keys = list(results_format['train'].keys())
        
        params = { 'ct_p': self.ct_p, 't_p': self.t_p, 'method_name': self.method_name, 'iter': iter}
        
        # wandb dictionary hyperparameters
        hps = dict( **self.ct_p, **self.t_p, **self.al_p, method_name=self.method_name, iter=iter)
        del hps['Dataset']
        hps['Model'] = hps['Model'].__class__.__name__
        
        # Pipe for the itra-process communication of the results
        parent_conn, child_conn = mp.Pipe()
        
        # if we are using multiple gpus
        if self.world_size > 1:
            logger.info(' => RUNNING DISTRIBUTED TRAINING')
            
            if (self.ct_p['wandb_logs']):
                logger.info(' => Logging in WandB!!!')
                params['wandb_p'] = wandb.init(project="AL_GTG", group="DDP", config=hps)
            
            # spawn the process
            mp.spawn(fn=train_ddp, args=(self.world_size, params, child_conn, ), nprocs=self.world_size, join=True) # type: ignore
            # obtain the results
            while parent_conn.poll(): train_recv, test_recv = parent_conn.recv()
            
            #wandb.finish()            
                
        else:
            logger.info(' => RUNNING TRAINING')
            # add the already created labeeld train dataloader
            
            if (self.ct_p['wandb_logs']):
                logger.info(' => Logging in WandB!!!')
                params['wandb_p'] = wandb.init(project="AL_GTG", config=hps)
                
            train_recv, test_recv = train(params)
            
            #wandb.finish()
        
        logger.info(' DONE\n')
        
        iter_train_results, iter_test_results = {}, {}
        for idx, metric in enumerate(train_recv): iter_train_results[train_res_keys[idx]] = metric
            
        for idx, metric in enumerate(test_recv): 
            iter_test_results[test_res_keys[idx]] = metric
            results_format['test'][test_res_keys[idx]].append(metric)
        
        logger.info(f'TESTING RESULTS -> {iter_test_results}')
        
        write_csv(
            task = self.ct_p['task'],
            ts_dir = self.ct_p['timestamp'],
            dataset_name = self.ct_p['dataset_name'],
            head = ['method', 'iter', 'lab_obs'] + test_res_keys,
            values = [self.method_name, self.ct_p['trial'], lab_obs] + list(iter_test_results.values())
        )
        
        save_train_val_curves(list(self.t_p['results_dict']['train'].keys()), iter_train_results, self.method_name,
                              self.ct_p['timestamp'], self.ct_p['dataset_name'], iter, self.ct_p['trial'])

        return iter_train_results
        
        
    
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
            Subset(self.dataset.transformed_trainset, self.labeled_indices),
            batch_size=self.t_p['batch_size'], shuffle=False, pin_memory=True
        )
        
        logger.info(' DONE\n')

        
        
    def run(self) -> Dict[str, List[float]]:
                
        results_format = copy.deepcopy(self.t_p['results_dict'])
        
        logger.info(f'----------------------- ITERATION {self.iter} / {self.al_p['al_iters']} -----------------------\n')
        
        self.train_results[self.iter] = self.train_evaluate_save(self.al_p['n_top_k_obs'], self.iter, results_format)
        
        
        # start of the loop
        while self.iter < self.al_p['al_iters']:

            self.iter += 1
            
            logger.info(f'----------------------- ITERATION {self.iter} / {self.al_p['al_iters']} -----------------------\n')
            
            logger.info(f' => Getting the sampled unalbeled subset for the current iteration...')
            samp_unlab_subset = self.get_samp_unlab_subset()
            logger.info(' DONE\n')
            
            logger.info(' START QUERY PROCESS\n')
            
            # run method query strategy
            idxs_new_labels, topk_idx_obs = self.query(samp_unlab_subset, self.al_p['n_top_k_obs'])
            
            d_labels = count_class_observation(self.dataset.classes, self.dataset.transformed_trainset, topk_idx_obs)
            logger.info(f' Number of observations per class added to the labeled set:\n {d_labels}\n')
            
            # Saving the tsne embeddings plot
            if self.method_name.split('_')[0] == 'GTG':
                # if we are performing GTG plot also the GTG predictions in the TSNE plot 
                self.save_tsne(samp_unlab_subset, idxs_new_labels, d_labels, self.iter, self.gtg_result_prediction)
            
            else: self.save_tsne(samp_unlab_subset, idxs_new_labels, d_labels, self.iter)

            # modify the datasets and dataloader and plot the tsne
            self.update_sets(topk_idx_obs)

            # iter + 1
            self.train_results[self.iter] = self.train_evaluate_save(self.iter * self.al_p['n_top_k_obs'], self.iter, results_format)
                
        epochs = len(self.train_results[1]['train_pred_loss'])
        
        # plotting the cumulative train results
        print_cumulative_train_results(list(self.t_p['results_dict']['train'].keys()), 
                                       self.train_results, self.method_name, epochs,
                                       self.ct_p['timestamp'], self.ct_p['dataset_name'], 
                                       self.ct_p['trial'])
        
        
        return results_format['test']