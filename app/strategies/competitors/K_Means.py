
from strategies.Strategies import Strategies

from torch.utils.data import DataLoader, Subset

import numpy as np
from sklearn.cluster import KMeans

from typing import Dict, Any, List

import logging
logger = logging.getLogger(__name__)



class K_Means(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], training_params: Dict[str, Any], LL: bool) -> None:
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__

        super().__init__(al_params, training_params, LL)
        
                           
        
    def apply_kmeans(self, n_clusters: int) -> np.ndarray:
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

        
        return closest_indices
    
    
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> List[int]:
                        
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True
        )
                            
        logger.info(' => Getting the unlabeled embeddings')
        self.embedds_dict = {'embedds': None, 'idxs': None}
        self.get_embeddings(self.unlab_train_dl, self.embedds_dict)
        logger.info(' DONE\n')
            
        logger.info(' => Top K extraction using k-means')
        topk_idx_obs = self.apply_kmeans(n_top_k_obs)
        logger.info(' DONE\n')
        
                
        return [self.embedds_dict['idxs'][id].item() for id in topk_idx_obs]
    