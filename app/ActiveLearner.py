
import wandb

from models.BBone_Module import Master_Model
from datasets_creation.Classification import Cls_Datasets, MySubset
from datasets_creation.Detection import Det_Dataset
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
import os

import logging
logger = logging.getLogger(__name__)



class ActiveLearner():
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any], strategy_name: str) -> None:
        
        self.iter = 1
        
        self.al_p: Dict[str, Any] = al_p
        self.ct_p: Dict[str, Any] = ct_p
        self.t_p: Dict[str, Any] = t_p
        
        self.device: torch.device = self.ct_p['device']
        self.model: Master_Model = self.ct_p['Master_Model']
        self.model = self.model.to(self.device)
        
        self.dataset: Cls_Datasets | Det_Dataset = self.ct_p['Dataset']
        
        self.ds_t_p = self.t_p[self.ct_p['dataset_name']]
        self.batch_size = self.ds_t_p['batch_size'][self.model.added_module_name]
                
        self.labeled_indices: List[int] = copy.deepcopy(self.dataset.labeled_indices)
        self.unlabeled_indices: List[int] = copy.deepcopy(self.dataset.unlabeled_indices)
        self.temp_unlab_pool = []

        self.train_results: Dict[str, Any] = {}
        
        self.labeled_subset = MySubset(self.dataset.transformed_trainset, self.labeled_indices)

        self.strategy_name = f'{self.model.name}_{strategy_name}' # define strategy name    
        self.best_check_filename: str = f'app/checkpoints/{self.ct_p['dataset_name']}/best_{self.strategy_name}'
        
        self.world_size: int = self.ct_p['gpus']
        
        self.path = f'results/{self.ct_p['timestamp']}/{self.ct_p['dataset_name']}/{self.ct_p['trial']}/{self.strategy_name}'
        create_method_res_dir(self.path)
        
        # save initial labeled images
        #self.save_labeled_images(self.labeled_indices)
        
        
        
    
    def save_labeled_images(self, new_labeled_idxs: List[int]) -> None:
        logger.info(f' => Iteration {self.iter} Method {self.strategy_name} - Saving the new labeled images for further visual analysis...')
        create_class_dir(self.path, self.iter, self.dataset.classes)
        for idx_top, (_, img, gt) in enumerate(Subset(self.dataset.non_transformed_trainset, new_labeled_idxs)): # type: ignore
            if self.ct_p['task'] != 'clf': 
                unique_labs = np.unique(np.array([labs[-1] for labs in gt]))
                for lab in unique_labs: 
                    save_image(img, f'{self.path}/new_labeled_images/{self.iter}/{self.dataset.classes[int(lab)]}/{idx_top}.png')
            else:
                save_image(img, f'{self.path}/new_labeled_images/{self.iter}/{self.dataset.classes[gt]}/{idx_top}.png')
        logger.info(' DONE\n')
        
        
    
    def get_samp_unlab_subset(self) -> Subset:
        if self.ct_p['task'] == 'clf':
            # set seed for reproducibility            
            seed = self.dataset.dataset_id * (self.ct_p['trial'] * self.al_p['al_iters'] + (self.iter - 1))
            set_seeds(seed)
            
            rand_perm = torch.randperm(len(self.unlabeled_indices)).tolist()
            self.rand_unlab_sample = [self.unlabeled_indices[idx] for idx in rand_perm[:self.ds_t_p['unlab_sample_dim']]]
            
            logger.info(f' SEED: {seed} - Last 10 permuted indices are: {rand_perm[-10:]}')
            unlab_perm_subset = Subset(self.dataset.non_transformed_trainset, self.rand_unlab_sample)
            
            # removing the whole observation sample fromt the unlabeled indices list
            for idx in self.rand_unlab_sample: self.unlabeled_indices.remove(idx) # - 10000
            
            logger.info(f' SEED: {seed} - With dataset indices: {unlab_perm_subset.indices[-10:]}')
            
            #reset the original seed
            set_seeds()
        
            return unlab_perm_subset
        else:
            return Subset(self.dataset.non_transformed_trainset, self.unlabeled_indices)
    
    
    def load_best_checkpoint(self):
        if dist.is_available():
            if self.world_size > 1: device = 'cuda:0'
            else: device = 'cuda' 
        else: device = 'cpu'

        checkpoint: Dict = torch.load(f'{self.best_check_filename}_{device}.pth.tar', map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
    
    
    def get_embeddings(self, dataloader: DataLoader, dict_to_modify: Dict[str, Any]) -> None:
        
        self.model.eval()

        # again no gradients needed
        with torch.inference_mode():
            
            for data in dataloader:
                if len(data) > 3: _, images, labels, _ = data
                else: _, images, labels = data
                
                if 'embedds' in dict_to_modify:
                    embed = self.model(images.to(self.device), mode='embedds')
                    dict_to_modify['embedds'] = torch.cat((dict_to_modify['embedds'], embed.squeeze()), dim=0)
                    
                if 'probs' in dict_to_modify:
                    outs = self.model(images.to(self.device), mode='probs')
                    dict_to_modify['probs'] = torch.cat((dict_to_modify['probs'], outs.squeeze().cpu()), dim=0)
                    
                # could be both LL (1 output) and GTG (2 outputs)
                if 'module_out' in dict_to_modify:
                    if self.model.added_module != None:
                        if self.model.added_module.__class__.__name__ == 'GTG_Module':
                            module_out = self.model(images.to(self.device), labels.to(self.device), mode='module_out')
                            dict_to_modify['module_out'] = torch.cat((dict_to_modify['module_out'], module_out[0][0].cpu().squeeze()), dim=0)
                        else:
                            module_out = self.model(images.to(self.device), mode='module_out')
                            dict_to_modify['module_out'] = torch.cat((dict_to_modify['module_out'], module_out.cpu().squeeze()), dim=0)
                    else:
                        raise AttributeError("Can't get the module_out if there is no additional module specified")    
                                        
                if 'labels' in dict_to_modify: dict_to_modify['labels'] = torch.cat((dict_to_modify['labels'], labels), dim=0)

    
    
    
    def save_tsne(self, samp_unlab_subset: Subset, idxs_new_labels: List[int], \
                  d_labels: Dict[str, int], al_iter: str, gtg_result_prediction = None) -> None:
        # plot the tsne graph for each iteration
        
        logger.info(' => Saving the TSNE embeddings plot with labeled, unlabeled and new labeled observations')
        
        dl_dict = dict(batch_size=self.batch_size, shuffle=False)
        unlab_train_dl = DataLoader(samp_unlab_subset, **dl_dict)
        lab_train_dl = DataLoader(self.labeled_subset, **dl_dict)
        
        lab_embedds_dict = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=self.device),
            'labels': torch.empty(0, dtype=torch.int8)
        }
        unlab_embedds_dict = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=self.device),
            'labels': torch.empty(0, dtype=torch.int8)
        }
            
        logger.info(' Getting the embeddings...')
        self.get_embeddings(lab_train_dl, lab_embedds_dict)
        self.get_embeddings(unlab_train_dl, unlab_embedds_dict)
        

        logger.info(' Plotting...')
        plot_new_labeled_tsne(
            lab_embedds_dict, unlab_embedds_dict,
            al_iter, self.strategy_name,
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
        
        params = { 'ct_p': self.ct_p, 't_p': self.t_p, 'strategy_name': self.strategy_name, 
                'iter': iter, 'labeled_subset': self.labeled_subset }
        
        # wandb dictionary hyperparameters
        hps = dict( **self.ct_p, **self.t_p, **self.al_p, strategy_name=self.strategy_name, iter=iter)
        del hps['Dataset']
        hps['Master_Model'] = hps['Master_Model'].__class__.__name__
        
        
        # if we are using multiple gpus
        if self.world_size > 1:
            
            os.environ["MASTER_ADDR"] = "ppv-gpu1"
            os.environ["MASTER_PORT"] = "16217"
            
            # Pipe for the itra-process communication of the results
            parent_conn, child_conn = mp.Pipe()
            
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
            
        
        logger.info(' DONE\n')
        
        iter_train_results, iter_test_results = {}, {}
        for idx, metrics in enumerate(train_recv): iter_train_results[train_res_keys[idx]] = metrics
            
        for idx, metrics in enumerate(test_recv): 
            iter_test_results[test_res_keys[idx]] = metrics
            results_format['test'][test_res_keys[idx]].append(metrics)
        
        logger.info(f'TESTING RESULTS -> {iter_test_results}')
        
        write_csv(
            task = self.ct_p['task'],
            ts_dir = self.ct_p['timestamp'],
            dataset_name = self.ct_p['dataset_name'],
            head = ['method', 'iter', 'lab_obs'] + test_res_keys,
            values = [self.strategy_name, self.ct_p['trial'], lab_obs] + list(iter_test_results.values())
        )
        
        save_train_val_curves(list(self.t_p['results_dict']['train'].keys()), iter_train_results, self.strategy_name,
                              self.ct_p['timestamp'], self.ct_p['dataset_name'], iter, self.ct_p['trial'])

        return iter_train_results
        
        
    
    def update_sets(self, unlab_samp_idxs: List[int], overall_topk: List[int]) -> None:        
        # save the new labeled images to further visual analysis
        #self.save_labeled_images(overall_topk)
        
        # Update the labeeld and unlabeled training set
        logger.info(' => Modifing the Subsets')
        
        
        sample_len = len(unlab_samp_idxs)
        # the sample have been already removed from the labeled set
        for idx in overall_topk: unlab_samp_idxs.remove(idx) # - 1000 = 9000
        self.temp_unlab_pool.extend(unlab_samp_idxs) # + 9000
        # extend with the overall_topk
        self.labeled_indices.extend(overall_topk) # + 1000
        
        
        # sanity check
        if len(list(set(self.unlabeled_indices) & set(self.labeled_indices))) == 0 and \
            len(list(set(self.temp_unlab_pool) & set(self.labeled_indices))) == 0 :
            logger.info(' Intersection between indices lists are EMPTY')
        else: 
            logger.exception('NON EMPTY INDICES INTERSECTION')
            raise Exception('NON EMPTY INDICES INTERSECTION')
        
        
        if len(self.unlabeled_indices) < sample_len and len(self.temp_unlab_pool) > 0:
            # reinsert all teh observations from the pool inside the original unlabeled pool
            self.unlabeled_indices.extend(self.temp_unlab_pool)
            self.temp_unlab_pool.clear() # empty the temp unlabeled pool list
        

        # generate the new labeled DataLoader
        self.labeled_subset = MySubset(self.dataset.transformed_trainset, self.labeled_indices)

        
        logger.info(f' New labeled_indices lenght: {len(self.labeled_indices)} - new unlabeled_indices lenght: {len(self.unlabeled_indices)}')
        
        logger.info(' DONE\n')

        
        
    def run(self) -> Dict[str, List[float]]:
                
        results_format = copy.deepcopy(self.t_p['results_dict'])
        
        logger.info(f'----------------------- ITERATION {self.iter} / {self.al_p['al_iters']} -----------------------\n')
        
        self.train_results[str(self.iter)] = self.train_evaluate_save(self.al_p['n_top_k_obs'], self.iter, results_format)
        
        
        # start of the loop
        while self.iter < self.al_p['al_iters']:

            self.iter += 1
            
            logger.info(f'----------------------- ITERATION {self.iter} / {self.al_p['al_iters']} -----------------------\n')
            
            logger.info(f' => Getting the sampled unalbeled subset for the current iteration...')
            samp_unlab_subset = self.get_samp_unlab_subset()
            logger.info(' DONE\n')
            
            logger.info(' START QUERY PROCESS\n')
            
            # run method query strategy
            idxs_new_labels, topk_idx_obs = self.query(samp_unlab_subset, self.al_p['n_top_k_obs']) # type: ignore
            
            d_labels = count_class_observation(self.dataset.classes, self.dataset.transformed_trainset, topk_idx_obs)
            logger.info(f' Number of observations per class added to the labeled set:\n {d_labels}\n')
            
            # Saving the tsne embeddings plot
            if len(self.strategy_name.split('_')) > 2 and self.strategy_name.split('_')[2] == 'GTG':
                # if we are performing GTG Ofline plot also the GTG predictions in the TSNE plot 
                self.save_tsne(samp_unlab_subset, idxs_new_labels, d_labels, str(self.iter), self.gtg_result_prediction) # type: ignore
            
            elif self.model.added_module_name == 'GTG_Module': self.save_tsne(samp_unlab_subset, idxs_new_labels, d_labels, str(self.iter))

            # modify the datasets and dataloader and plot the tsne
            self.update_sets(list(samp_unlab_subset.indices), topk_idx_obs)

            # iter + 1
            self.train_results[str(self.iter)] = self.train_evaluate_save( self.al_p['init_lab_obs'] + ((self.iter - 1) * self.al_p['n_top_k_obs']), self.iter, results_format)
                
        epochs = len(self.train_results['1']['train_pred_loss'])
        
        # plotting the cumulative train results
        print_cumulative_train_results(list(self.t_p['results_dict']['train'].keys()), 
                                       self.train_results, self.strategy_name, epochs,
                                       self.ct_p['timestamp'], self.ct_p['dataset_name'], 
                                       self.ct_p['trial'])
        
        
        return results_format['test']