
from TrainEvaluate import TrainEvaluate
from utils import write_csv, entropy
from cifar10 import UniqueShuffle

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from termcolor import colored
import copy



class GTG_Strategy(TrainEvaluate):
    
    def __init__(self, al_params, our_methods_params):
        super().__init__(al_params)
        
        self.method_name = self.__class__.__name__
        self.params = our_methods_params 


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



    # correct
    def get_X(self, len_samp_unlab_embeds):
        
        self.X = torch.zeros(len(self.lab_train_ds) + len_samp_unlab_embeds, self.n_classes, dtype=torch.float32).to(self.device)
        
        for idx, (_, _, label) in enumerate(self.lab_train_ds): self.X[idx][label] = 1
        for idx in range(len(self.lab_train_ds), len(self.lab_train_ds) + len_samp_unlab_embeds):
            for label in range(self.n_classes):
                self.X[idx][label] = 1 / self.n_classes
        


    def gtg(self, tol, max_iter, indices):
        err = float('Inf')
        i = 0
        
        while err > tol and i < max_iter:
            X_old = copy.deepcopy(self.X) #torch.clone(self.X)
            self.X *= torch.mm(self.A, self.X)

                
            self.X /= torch.sum(self.X, dim=1, keepdim=True)
        
            iter_entropy = entropy(self.X).to(self.device) # both labeled and sample unlabeled
            # I have to map only the sample_unlabeled to the correct position
            
            #for idx, unlab_ent_val in tqdm(enumerate(iter_entropy[len(self.lab_train_ds):]), total = len(iter_entropy) - len(self.lab_train_ds), desc = f'Computing the derivatives of iteration {i}', leave = False):
            for idx, unlab_ent_val in enumerate(iter_entropy[len(self.lab_train_ds):]):
                # I iterate only the sampled unlabeled one
                
                if(i != self.params['gtg_max_iter'] - 1): self.entropy_pairwise_der[indices[idx]][i] = unlab_ent_val

                if i != 0: self.entropy_pairwise_der[indices[idx]][i - 1] = self.entropy_pairwise_der[indices[idx]][i - 1] - unlab_ent_val 
    

            err = torch.norm(self.X - X_old)
            i += 1



    def run(self, al_iters, epochs, n_top_k_obs):
        results = {}
        
        for n_splits in self.params['list_n_samples']:
            
            # deeper copy of the original datasets and labeled train dataloader
            
                    
            print(colored(f'----------------------- WORKING WITH {n_splits} UNLABELED SPLITS -----------------------\n', 'green'))
                    
            iter = 0

            results[n_splits] = { 'test_accuracy': [], 'test_loss': [] }
                
            # iter = 0            
            print(colored(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n', 'blue'))
            self.reintialize_model()
            self.fit(epochs, self.lab_train_dl, self.method_name) # train in the labeled observations
            
            test_accuracy, test_loss = self.test_AL()
                
            write_csv(
                ts_dir=self.timestamp,
                head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                values = [self.method_name, iter, n_splits, test_accuracy, test_loss]
            )
                
            results[n_splits]['test_accuracy'].append(test_accuracy)
            results[n_splits]['test_loss'].append(test_loss)
                     
                     
            # start of the loop   
            while len(self.unlab_train_ds) > 0 and iter < al_iters:
                print(colored(f'----------------------- ITERATION {iter + 1} / {al_iters} -----------------------\n', 'blue')) 
                
                # Obtaining the updated batchsize
                iter_batch_size = len(self.unlab_train_ds) // n_splits
                                
                self.unlab_train_dl = DataLoader(
                    self.unlab_train_ds,
                    batch_size=iter_batch_size,
                    sampler=UniqueShuffle(self.unlab_train_ds),
                    num_workers=1,
                    pin_memory=True
                )
                
                # index are referred to slf.train_ds
                # and are consistent emebdding creation and the read of the indices

                
                self.labeled_embeddings = self.get_embeddings('Labeled', self.lab_train_dl)
                self.unlab_embeddings = self.get_embeddings('Unlabeled', self.unlab_train_dl)
                
                
                # at each AL round I reinitialize the entropy_pairwise_der since I have to decide at each step what observations I want to move
                self.entropy_pairwise_der = torch.zeros((len(self.train_ds), self.params['gtg_max_iter'] - 1)).to(self.device)

                # for each split
                pbar = tqdm(enumerate(self.unlab_train_dl), total=len(self.unlab_train_dl), leave=True)

                # idx -> indices of the split
                # indices -> are the list of indices for the given batch which ARE NOT CONSISTENT SINCE ARE REFERRED TO THE INDEX OF THE ORIGINAL DATASET
                for idx, (indices, _, _) in pbar:
                                        
                    pbar.set_description(f'WORKING WITH UNLABELED SAMPLE # {idx + 1}')
                                
                    self.get_A(self.unlab_embeddings[idx * iter_batch_size : (idx + 1) * iter_batch_size])
                    self.get_X(iter_batch_size)
                    self.gtg(self.params['gtg_tol'], self.params['gtg_max_iter'], indices)
                    
                    self.clear_memory()
                    
                    

                # mean of the entropy derivate 
                #print(torch.sum(self.entropy_pairwise_der, dim = 1, dtype=torch.float32))
                overall_topk = torch.topk((torch.sum(self.entropy_pairwise_der, dim = 1, dtype=torch.float32) / self.entropy_pairwise_der.shape[1]), n_top_k_obs)
                
                #print(overall_topk.values[:10])
                # HO TUTTE LE DERIVATE DELLE ENTROPIE MOLTO SIMILI QUESTO VA AD INDICARE CHE I MIEI ESEMPI SONO SIMILI PER QUANTO RIGUARDA LA LORO DIFFICOLTA'
                
                #overall_topk.indices -> è riferito agli indici della matrice entropy_pairwise_der
                
                self.get_new_dataloaders(overall_topk.indices.tolist())
                
                # iter + 1
                self.reintialize_model()
                self.fit(epochs, self.lab_train_dl, self.method_name) # train in the labeled observations
                
                test_accuracy, test_loss = self.test_AL()
                
                write_csv(
                    ts_dir=self.timestamp,
                    head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                    values = [self.method_name, iter + 1, n_splits, test_accuracy, test_loss]
                )
                
                results[n_splits]['test_accuracy'].append(test_accuracy)
                results[n_splits]['test_loss'].append(test_loss)
                
                        
                iter += 1
        
        return results