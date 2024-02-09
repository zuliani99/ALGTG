
from TrainEvaluate import TrainEvaluate
from utils import save_train_val_curves, write_csv, entropy
from CIFAR10 import UniqueShuffle

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import copy



class GTG_Strategy(TrainEvaluate):
    
    def __init__(self, al_params, our_methods_params, LL):
        super().__init__(al_params, LL)
                
        self.method_name = self.__class__.__name__
        self.params = our_methods_params 
        self.LL = LL
        
        #self.weights = torch.flip(torch.linspace(1, 0.1, self.params['gtg_max_iter'] - 1), [0]).to(self.device)


    def clear_memory(self):
        del self.X
        del self.A
        torch.cuda.empty_cache()


    def get_A(self, samp_unlab_embeddings):
        
        normalized_embedding = F.normalize(
            torch.cat((self.labeled_embeddings, samp_unlab_embeddings)).to(self.device)
        , dim=-1).to(self.device)
        
        self.A = F.relu(
            torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device)
        ).fill_diagonal_(0)
        
        del normalized_embedding



    def get_X(self, target_lab_obs, len_samp_unlab_embeds):
        
        self.X = torch.zeros(len(self.lab_train_subset) + len_samp_unlab_embeds, self.n_classes, dtype=torch.float32, device=self.device)

        for idx, label in enumerate(target_lab_obs): self.X[idx][label] = 1
        
        for idx in range(len(self.lab_train_subset), len(self.lab_train_subset) + len_samp_unlab_embeds):
            for label in range(self.n_classes):
                self.X[idx][label] = 1 / self.n_classes
        
        

    def gtg(self, tol, max_iter, indices):
        err = float('Inf')
        i = 0
        
        while err > tol and i < max_iter:
            X_old = copy.deepcopy(self.X)
            self.X *= torch.mm(self.A, self.X)

            self.X /= torch.sum(self.X, dim=1, keepdim=True)
        
            iter_entropy = entropy(self.X).to(self.device) # both labeled and sample unlabeled
            # I have to map only the sample_unlabeled to the correct position
                
            for idx, unlab_ent_val in zip(indices, iter_entropy[len(self.lab_train_subset):]):
                # I iterate only the sampled unlabeled one
                
                if (i != self.params['gtg_max_iter'] - 1): self.entropy_pairwise_der[idx][i] = unlab_ent_val

                if (i != 0): self.entropy_pairwise_der[idx][i - 1] = self.entropy_pairwise_der[idx][i - 1] - unlab_ent_val 
    

            err = torch.norm(self.X - X_old)
            i += 1



    def run(self, al_iters, epochs, n_top_k_obs):
        results = {}
        
        for n_splits in self.params['list_n_samples']:
                                
            print(f'----------------------- WORKING WITH {n_splits} UNLABELED SPLITS -----------------------\n')
                    
            iter = 0

            results[n_splits] = { 'test_accuracy': [], 'test_loss': [] }
                
            # iter = 0            
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
            
            
            # reset the indices to the original one
            self.original_trainset.lab_train_idxs = self.lab_train_subset.indices
            self.reintialize_model()
            
            
            train_results = self.train_evaluate(epochs, self.lab_train_dl, self.method_name) # train in the labeled observations
            
            save_train_val_curves(train_results, self.timestamp, iter)
            
            test_accuracy, test_loss = self.test()
                
            write_csv(
                ts_dir=self.timestamp,
                head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                values = [self.method_name, iter, n_splits, test_accuracy, test_loss]
            )
                
            results[n_splits]['test_accuracy'].append(test_accuracy)
            results[n_splits]['test_loss'].append(test_loss)
                     
                     
            # start of the loop   
            while len(self.unlab_train_subset) > 0 and iter < al_iters:
                print(f'----------------------- ITERATION {iter + 1} / {al_iters} -----------------------\n')
                
                # Obtaining the actual batchsize for the unlabeled observations
                iter_batch_size = len(self.unlab_train_subset) // n_splits
                                
                self.unlab_train_dl = DataLoader(
                    self.unlab_train_subset,
                    batch_size=iter_batch_size,
                    sampler=UniqueShuffle(self.unlab_train_subset),
                    
                    pin_memory=True
                )
                
                print(' => Getting the labeled and unlabeled embeddings')
                self.labeled_embeddings, target_lab_obs = self.get_embeddings(self.lab_train_dl)
                self.unlab_embeddings, _ = self.get_embeddings(self.unlab_train_dl)
                print(' DONE\n')
                
                # at each AL round I reinitialize the entropy_pairwise_der since I have to decide at each step what observations I want to move
                self.entropy_pairwise_der = torch.zeros((len(self.original_trainset), self.params['gtg_max_iter'] - 1), device=self.device)

                # for each split

                # split_idx -> indices of the split
                # indices -> are the list of indices for the given batch which ARE CONSISTENT SINCE ARE REFERRED TO THE INDEX OF THE ORIGINAL DATASET
                for split_idx, (indices, _, _) in enumerate(self.unlab_train_dl):
                    print(f' => Running GTG for split {split_idx}')
                    self.get_A(self.unlab_embeddings[split_idx * iter_batch_size : (split_idx + 1) * iter_batch_size])
                    self.get_X(target_lab_obs, indices.shape[0])
                    self.gtg(self.params['gtg_tol'], self.params['gtg_max_iter'], indices)
                    
                    self.clear_memory()
                    print(' DONE\n')
                    
                    

                # mean of the entropy derivate 
                #print(torch.sum(self.entropy_pairwise_der, dim = 1, dtype=torch.float32))
                #overall_topk = torch.topk((torch.sum(self.entropy_pairwise_der, dim = 1, dtype=torch.float32) / self.entropy_pairwise_der.shape[1]), n_top_k_obs)
                
                
                # weighted average (* self.weights)
                overall_topk = torch.topk(torch.mean(self.entropy_pairwise_der, dim = 1), n_top_k_obs)
                                
                #overall_topk.indices -> referred to the amtrix indices of entropy_pairwise_der
                
                print(' => Modifing the Subsets and Dataloader')
                self.get_new_dataloaders(overall_topk.indices.tolist())
                print(' DONE\n')
                
                # iter + 1
                self.reintialize_model()
                train_results = self.train_evaluate(epochs, self.lab_train_dl, self.method_name) # train in the labeled observations
                
                save_train_val_curves(train_results, self.timestamp, iter + 1)
                
                test_accuracy, test_loss = self.test()
                
                write_csv(
                    ts_dir=self.timestamp,
                    head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                    values = [self.method_name, iter + 1, n_splits, test_accuracy, test_loss]
                )
                
                results[n_splits]['test_accuracy'].append(test_accuracy)
                results[n_splits]['test_loss'].append(test_loss)
                
                        
                iter += 1
        
        return results