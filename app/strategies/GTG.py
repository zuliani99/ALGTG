
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
            str_tresh_strat = f'{gtg_p["am_ts"]}_{gtg_p["am_t"]}' if gtg_p["am_ts"] != 'mean' else 'ts-mean'            
            strategy_name = f'{self.__class__.__name__}_{gtg_p["am_s"]}_{gtg_p["am"]}_{str_tresh_strat}_es-{gtg_p["e_s"]}'
        else: strategy_name = f'{self.__class__.__name__}_{gtg_p["am_s"]}_{gtg_p["am"]}_es-{gtg_p["e_s"]}'
                
        super().__init__(ct_p, t_p, al_p, strategy_name)
        
        if self.model.added_module != None: self.model.added_module.define_A_function(gtg_p["am"])
        
    
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        query_bs = self.batch_size//2
           
        unlab_train_dl = DataLoader(sample_unlab_subset, batch_size=query_bs, shuffle=False, pin_memory=True)
        labelled_subset = Subset(self.dataset.unlab_train_ds, self.labelled_indices)
        lab_train_dl = DataLoader(
            dataset=labelled_subset, batch_size=query_bs,
            sampler=RandomSampler(labelled_subset, num_samples=len(sample_unlab_subset)), 
            #random sample with replacement, each batch has different set of labelled observation drown ad random from the entire set
        )
        
        self.load_best_checkpoint()
        
        pred_entropies = torch.empty(0, dtype=torch.float32, device=self.device)
        true_entropies = torch.empty(0, dtype=torch.float32, device=self.device)
        
        logger.info(' => Running GTG in inference mode...')
        
        self.model.eval()
        with torch.inference_mode():
            
            for (_, lab_images, lab_labels), (_, unlab_images, unlab_labels) in zip(lab_train_dl, unlab_train_dl):
                                            
                lab_outs, lab_embedds = self.model.backbone(lab_images.to(self.device))
                lab_features = self.model.backbone.get_features()
                unlab_outs, unlab_embedds = self.model.backbone(unlab_images.to(self.device))
                unlab_features = self.model.backbone.get_features()

                (y_pred, y_true) ,_ = self.model.added_module(
                    features=[torch.cat((lab_feature, unlab_feature), dim=0) for lab_feature, unlab_feature in zip(lab_features, unlab_features)], 
                    embedds=torch.cat((lab_embedds, unlab_embedds), dim=0), 
                    outs=torch.cat((lab_outs, unlab_outs), dim=0),
                    labels=torch.cat((lab_labels, unlab_labels), dim=0)
                )
                
                # save only the unalbelled entropies
                pred_entropies = torch.cat((pred_entropies, y_pred[query_bs:]), dim=0) 
                true_entropies = torch.cat((true_entropies, y_true[query_bs:]), dim=0)
                
        logger.info(' DONE\n')
        
        
        ############# DEBUG PORPUSE ############# 
        top_k_pred = torch.topk(pred_entropies, n_top_k_obs).indices.to(self.device)
        top_k_true = torch.topk(true_entropies, n_top_k_obs).indices.to(self.device)
        
        logger.info(torch.topk(pred_entropies, n_top_k_obs))
        logger.info(torch.topk(true_entropies, n_top_k_obs))
        
        intersection_count = len(set(top_k_pred.cpu().tolist()) & set(top_k_true.cpu().tolist()))
        intersection_ratio = intersection_count / len(top_k_pred)
        
        logger.info(f' => Intersection ratio between predicted and true entropies: {intersection_ratio}')
                
        #mse_true_entr = ((pred_entropies[top_k_true].double() - true_entropies[top_k_true].double()) ** 2).mean()
        #mse_pred_entr = ((pred_entropies[top_k_pred].double() - true_entropies[top_k_pred].double()) ** 2).mean()
        
        logger.info(f'MSE true_entr: {F.mse_loss(pred_entropies[top_k_true], true_entropies[top_k_true])}')
        logger.info(f'MSE pred_entr: {F.mse_loss(pred_entropies[top_k_pred], true_entropies[top_k_pred])}')
        ############# DEBUG PORPUSE ############# 
        
        
        logger.info(f' => Extracting the Top-k unlabelled observations')
        overall_topk = torch.topk(pred_entropies, n_top_k_obs).indices.tolist()
        logger.info(' DONE\n')
        
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]
        