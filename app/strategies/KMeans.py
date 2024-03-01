
from TrainEvaluate import TrainEvaluate
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader, Subset

import torch

class KMeans(TrainEvaluate):
    
    def __init__(self, al_params, LL):
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.LL = LL
                           

    def apply_kmeans(self, n_clusters):
        self.unlab_embeddings = self.unlab_embeddings.cpu().numpy()
        
        cluster_learner = KMeans(n_clusters=n_clusters)
        cluster_learner.fit(self.unlab_embeddings)
        
        cluster_idxs = cluster_learner.predict(self.unlab_embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dist = (self.unlab_embeddings - centers)**2
        dist = dist.sum(dim=1)
        
        closest_indices = torch.stack([
            torch.arange(self.unlab_embeddings.size(0))[cluster_idxs == i][
                dist[cluster_idxs == i].argmin()
            ]
            for i in range(n_clusters)
        ])
        
        return closest_indices
    

    def run(self, al_iters, epochs, unlab_sample_dim, n_top_k_obs):
        
        iter = 1
        
        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
        
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
        
        self.train_evaluate_save(epochs, n_top_k_obs, iter, results)
        
        # start of the loop
        while len(self.unlabeled_indices) > 0 and iter < al_iters:
            iter += 1
            
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
            
            sample_unlab_subset = Subset(
                self.non_transformed_trainset,
                self.get_unlabebled_samples(unlab_sample_dim, iter)
            )
                        
            self.unlab_train_dl = DataLoader(
                sample_unlab_subset, batch_size=self.batch_size,
                shuffle=True, pin_memory=True
            )
                            
            print(' => Getting the unlabeled embeddings')
            unlabeled_indices, self.unlab_embeddings, _  = self.get_embeddings(self.unlab_train_dl)
            print(' DONE\n')
            
            print(' => Top K extraction using k-means')
            indices_to_convert = self.apply_kmeans(n_top_k_obs)
            print(' DONE\n')
                        
            # modify the datasets and dataloader
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders([unlabeled_indices[id].item() for id in indices_to_convert])
            print(' DONE\n')
            
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
            
        self.remove_model_opt()
            
        return results
        