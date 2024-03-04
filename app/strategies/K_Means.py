
from strategies.Strategies import Strategies

from torch.utils.data import DataLoader

import numpy as np
from sklearn.cluster import KMeans


class K_Means(Strategies):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
                           
        
    def apply_kmeans(self, n_clusters):
        unlab_embeddings = self.embedds_dict['embedds'].cpu().numpy()
        
        cluster_learner = KMeans(n_clusters=n_clusters)
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
    
    
    def query(self, sample_unlab_subset, n_top_k_obs):
                        
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True
        )
                            
        print(' => Getting the unlabeled embeddings')
        self.embedds_dict = {'embedds': None, 'idxs': None}
        self.get_embeddings(self.unlab_train_dl, self.embedds_dict)
        print(' DONE\n')
            
        print(' => Top K extraction using k-means')
        topk_idx_obs = self.apply_kmeans(n_top_k_obs)
        print(' DONE\n')
        
        self.clear_cuda_variables([self.embedds_dict])
        
        return [self.embedds_dict['idxs'][id].item() for id in topk_idx_obs]
    