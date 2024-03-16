
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Subset

from strategies.Strategies import Strategies

from typing import Dict, Any, List


class ProbCover(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], LL: bool, al_iters: int, n_top_k_obs: int, unlab_sample_dim: int) -> None:
        super().__init__(al_params, LL, al_iters, n_top_k_obs, unlab_sample_dim)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.delta = 0.5


    # SHOULD BE OK
    def construct_graph(self):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        xs, ys, ds = [], [], []
        print(f'Start constructing graph using delta={self.delta}')
        # distance computations are done in GPU
        for i in range(len(self.rel_features) // self.batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = self.rel_features[i * self.batch_size: (i + 1) * self.batch_size]
            dist = torch.cdist(cur_feats, self.rel_features)
            mask = dist < self.delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + self.batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        print(f'Finished constructing graph using delta={self.delta}')
        print(f'Graph contains {len(df)} edges.')
        return df


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> List[int]:
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True
        )
        
        print(' => Getting the labeled and unlabeled embedding')
        self.lab_embedds_dict, self.unlab_embedds_dict = {'embeds': None, 'idxs': None}, {'embeds': None, 'idxs': None}
        self.get_embeddings(self.lab_train_dl, self.lab_embedds_dict)
        self.get_embeddings(self.unlab_train_dl, self.unlab_embedds_dict)

        self.relevant_indices = torch.cat(self.lab_embedds_dict['idxs'], self.unlab_embedds_dict['idxs']).cpu().tolist()
        self.rel_features = torch.cat(self.lab_embedds_dict['embeds'], self.unlab_embedds_dict['embeds']).to(self.device)


        self.construct_graph()


        print(f'Start selecting {n_top_k_obs} samples.')
        selected = []
        # removing incoming edges to all covered samples from the existing labeled set
        edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        covered_samples = self.graph_df.y[edge_from_seen].unique()
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        for i in range(n_top_k_obs):
            coverage = len(covered_samples) / len(self.relevant_indices)
            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices))
            print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
            cur = degrees.argmax()
            # cur = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]
        #remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        #print(f'Finished the selection of {len(activeSet)} samples.')
        #print(f'Active set is {activeSet}')
        #return activeSet, remainSet
        return activeSet

        #sample from all the set of incides......check if it will sample only from the unalbeled set, read the paper