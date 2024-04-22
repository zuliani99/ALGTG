
from ActiveLearner import ActiveLearner

from torch.utils.data import DataLoader, Subset
import torch

import numpy as np
from sklearn.cluster import KMeans

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)



class K_Means(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any]) -> None:
        
        super().__init__(ct_p, t_p, al_p, self.__class__.__name__)
        
                           
        
    def apply_kmeans(self, n_clusters: int) -> List[int]:
        unlab_embeddings = self.embedds_dict['embedds'].cpu().numpy()
        
        cluster_learner = KMeans(n_clusters=n_clusters, n_init='auto')
        cluster_learner.fit(unlab_embeddings)
        
        cluster_idxs = cluster_learner.predict(unlab_embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]

        dis = (unlab_embeddings - centers)**2
        dis = dis.sum(axis=1)
        closest_indices = np.array([
            np.arange(unlab_embeddings.shape[0])[cluster_idxs==i][
                dis[cluster_idxs==i].argmin()
            ] for i in range(n_clusters)
        ])

        
        return closest_indices.tolist()
    
    
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
                        
        unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.ds_t_p['batch_size'],
            shuffle=False, pin_memory=True
        )
                            
        logger.info(' => Getting the unlabeled embeddings')
        self.embedds_dict = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=self.device),
            'idxs': torch.empty(0, dtype=torch.int8)
        }
        
        self.load_best_checkpoint()
        
        self.get_embeddings(unlab_train_dl, self.embedds_dict)
        logger.info(' DONE\n')
            
        logger.info(' => Top K extraction using k-means')
        topk_idx_obs = self.apply_kmeans(n_top_k_obs)
        logger.info(' DONE\n')
        
                
        return topk_idx_obs, [int(self.embedds_dict['idxs'][id].item()) for id in topk_idx_obs]
    