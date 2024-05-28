
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, RandomSampler

from ActiveLearner import ActiveLearner

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class GTG(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any], gtg_p) -> None:
                
        if gtg_p["am_ts"] != None:
            str_tresh_strat = f'{gtg_p["am_ts"][gtg_p["id_am_ts"]]}_{gtg_p["am_t"]}' if gtg_p["am_ts"][gtg_p["id_am_ts"]] != 'mean' else 'ts-mean'            
            strategy_name = f'{self.__class__.__name__}_{gtg_p["am_s"]}_{gtg_p["am"][gtg_p["id_am"]]}_{str_tresh_strat}_es-{gtg_p["e_s"]}'
        else: strategy_name = f'{self.__class__.__name__}_{gtg_p["am_s"]}_{gtg_p["am"][gtg_p["id_am"]]}_es-{gtg_p["e_s"]}'
                
        super().__init__(ct_p, t_p, al_p, strategy_name)
        self.perc_labelled_batch: int = gtg_p["plb"]
        if self.model.added_module_name != None: self.model.added_module.define_idx_params(gtg_p["id_am_ts"], gtg_p["id_am"]) # -> for GTG module only
        
        
        
    '''def query_2(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        query_bs = int(self.batch_size * self.perc_labelled_batch)
        dict_dl = dict(batch_size=query_bs, shuffle=False, pin_memory=True)
        unlab_train_dl = DataLoader(sample_unlab_subset, **dict_dl)
        lab_train_dl = DataLoader(Subset(self.dataset.unlab_train_ds, self.labelled_indices), **dict_dl) 
        
        tot_lab_features = [
            torch.empty((0, 64, 32, 32), device=self.device),
            torch.empty((0, 128, 16, 16), device=self.device),
            torch.empty((0, 256, 8, 8), device=self.device),
            torch.empty((0, 512, 4, 4), device=self.device),
        ]
        tot_unlab_features = [lab_feature.clone() for lab_feature in tot_lab_features]
        
        tot_lab_outs = torch.empty(0, device=self.device)
        tot_unlab_outs = tot_lab_outs.clone()
        
        tot_lab_labels = tot_lab_outs.clone()
        tot_unlab_labels = tot_lab_outs.clone()
        
        tot_lab_embedds = torch.empty((0, self.model.backbone.get_embedding_dim()), device=self.device)
        tot_unlab_embedds = tot_lab_embedds.clone()

        self.load_best_checkpoint()
        self.model.eval()
        with torch.inference_mode():
            
            for _, lab_images, lab_labels in lab_train_dl:
                lab_outs, lab_embedds = self.model.backbone(lab_images.to(self.device))
                lab_features = self.model.backbone.get_features()
                
                for idx, lab_feature in enumerate(lab_features):
                    tot_lab_features[idx] = torch.cat((tot_lab_features[idx], lab_feature[idx].unsqueeze(dim=0)), dim=0)
                
                tot_lab_embedds=torch.cat((tot_lab_embedds, lab_embedds), dim=0), 
                tot_lab_outs=torch.cat((tot_lab_outs, lab_outs), dim=0),
                tot_lab_labels=torch.cat((tot_lab_labels, lab_labels), dim=0)
                
            for _, unlab_images, unlab_labels in unlab_train_dl:
                unlab_outs, unlab_embedds = self.model.backbone(unlab_images.to(self.device))
                unlab_features = self.model.backbone.get_features()
                
                for idx, unlab_feature in enumerate(unlab_features):
                    tot_unlab_features[idx] = torch.cat((tot_unlab_features[idx], unlab_feature[idx].unsqueeze(dim=0)), dim=0)
            
                tot_unlab_embedds=torch.cat((tot_unlab_embedds, unlab_embedds), dim=0), 
                tot_unlab_outs=torch.cat((tot_unlab_outs, unlab_outs), dim=0),
                tot_unlab_labels=torch.cat((tot_unlab_labels, unlab_labels), dim=0)
                
            (y_pred, _) ,_ = self.model.added_module(
                features=[torch.cat((lab_feature, unlab_feature), dim=0) for lab_feature, unlab_feature in zip(tot_lab_features, tot_unlab_features)], 
                embedds=torch.cat((tot_lab_embedds, tot_unlab_embedds), dim=0), 
                outs=torch.cat((tot_lab_outs, tot_unlab_outs), dim=0),
                labels=torch.cat((tot_lab_labels, tot_unlab_labels), dim=0),
                labelled_dim = len(tot_lab_labels)
            )
        
        logger.info(f' => Extracting the Top-k unlabelled observations')
        overall_topk = torch.topk(y_pred[len(tot_lab_labels):], n_top_k_obs).indices.tolist()

        logger.info(' DONE\n')
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]'''
        
        

    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        query_bs = int(self.batch_size * self.perc_labelled_batch)
           
        unlab_train_dl = DataLoader(sample_unlab_subset, batch_size=query_bs, shuffle=False, pin_memory=True)
        labelled_subset = Subset(self.dataset.unlab_train_ds, self.labelled_indices)
        lab_train_dl = DataLoader(
            dataset=labelled_subset, batch_size=query_bs,
            sampler=RandomSampler(labelled_subset, num_samples=len(sample_unlab_subset)), 
            #random sample with replacement, each batch has different set of labelled observation drown ad random from the entire set
        )        
        
        pred_entropies = torch.empty(0, dtype=torch.float32, device=self.device)
        true_entropies = torch.empty(0, dtype=torch.float32, device=self.device)
        
        logger.info(' => Running GTG in inference mode...')
        
        self.load_best_checkpoint()
        self.model.eval()
        with torch.inference_mode():
            
            for (_, lab_images, lab_labels), (_, unlab_images, unlab_labels) in zip(lab_train_dl, unlab_train_dl):
                                            
                lab_outs, lab_embedds = self.model.backbone(lab_images.to(self.device))
                lab_features = self.model.backbone.get_features()
                unlab_outs, unlab_embedds = self.model.backbone(unlab_images.to(self.device))
                unlab_features = self.model.backbone.get_features()

                (y_pred, y_true) ,_ = self.model.added_module(
                    features=[
                        torch.cat((lab_feature, unlab_feature), dim=0) for lab_feature, unlab_feature in zip(lab_features, unlab_features)
                    ], 
                    embedds=torch.cat((lab_embedds, unlab_embedds), dim=0), 
                    outs=torch.cat((lab_outs, unlab_outs), dim=0),
                    labels=torch.cat((lab_labels, unlab_labels), dim=0)
                )
                
                # save only the unalbelled entropies
                pred_entropies = torch.cat((pred_entropies, y_pred[query_bs:]), dim=0) 
                true_entropies = torch.cat((true_entropies, y_true[query_bs:]), dim=0)
                
        logger.info(' DONE\n')
        
        ############# DEBUG PORPUSE ############# 
        top_k_pred = torch.topk(pred_entropies, n_top_k_obs).indices
        top_k_true = torch.topk(true_entropies, n_top_k_obs).indices
        
        logger.info(torch.topk(pred_entropies, n_top_k_obs))
        logger.info(torch.topk(true_entropies, n_top_k_obs))
        
        intersection_count = len(set(top_k_pred.cpu().tolist()) & set(top_k_true.cpu().tolist()))
        intersection_ratio = intersection_count / len(top_k_pred)
        
        logger.info(f' => Intersection ratio between predicted and true entropies: {intersection_ratio}')
        
        logger.info(f'MSE true_entr: {F.mse_loss(pred_entropies[top_k_true], true_entropies[top_k_true])}')
        logger.info(f'MSE pred_entr: {F.mse_loss(pred_entropies[top_k_pred], true_entropies[top_k_pred])}')
        ############# DEBUG PORPUSE ############# 
        
        
        logger.info(f' => Extracting the Top-k unlabelled observations')
        overall_topk = top_k_pred.tolist()
        ######################################################
        #overall_topk = top_k_true.tolist() # -> DEBUB PORPUSE
        ######################################################
        logger.info(' DONE\n')
        
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]
        
        
        
        
# FOR MLP MODULE ONLY
'''unlab_train_dl = DataLoader(
            sample_unlab_subset,
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

        self.load_best_checkpoint()
        
        pred_entropies = torch.empty(0, dtype=torch.float32, device=self.device)
        true_entropies = torch.empty(0, dtype=torch.float32, device=self.device)
        
        logger.info(' => Running GTG in inference mode...')
        
        self.model.eval()
        with torch.inference_mode():
            for _, unlab_images, unlab_labels in unlab_train_dl:
                unlab_outs, unlab_embedds = self.model.backbone(unlab_images.to(self.device))
                unlab_features = self.model.backbone.get_features()
                (y_pred, y_true) ,_ = self.model.added_module(
                    features=unlab_features, embedds=unlab_embedds, outs=unlab_outs, labels=unlab_labels)
                
                pred_entropies = torch.cat((pred_entropies, y_pred), dim=0) 
                true_entropies = torch.cat((true_entropies, y_true), dim=0)
                
        logger.info(' DONE\n')'''
        
        
        # FOR LSTM MODULE
'''self.unlab_train_dl = DataLoader(sample_unlab_subset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
                
        logger.info(' => Evaluating unlabelled observations')
        embeds_dict = { 'module_out': torch.empty(0, dtype=torch.float32) }
        
        self.load_best_checkpoint()
        
        self.get_embeddings(self.unlab_train_dl, embeds_dict)
        
        logger.info(f' => Extracting the Top-k unlabelled observations')
        overall_topk = torch.topk(embeds_dict["module_out"], n_top_k_obs, largest=False).indices.tolist()
        logger.info(' DONE\n')
        
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]'''