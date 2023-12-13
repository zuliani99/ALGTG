
from utils import write_csv, entropy
from cifar10 import CIFAR10

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
from termcolor import colored



class GTG():
    
    def __init__(self, Main_AL_class, our_methods_params):
        self.method_name = 'AL_GTG'
        self.Main_AL_class = Main_AL_class
        
        self.params = our_methods_params
   
        self.lab_train_dl = self.Main_AL_class.lab_train_dl
        
        self.lab_train_ds = self.Main_AL_class.lab_train_ds
        self.unlab_train_ds = self.Main_AL_class.unlab_train_ds
                


    def clear_memory(self):
        del self.X
        del self.A
        torch.cuda.empty_cache()



    def get_unlabeled_samples(self, n_split):
        self.unlab_samp_list = []

        new_unlab_train_ds = self.unlab_train_ds
        sampled_unlab_size = int(len(new_unlab_train_ds) // n_split)
        
        print('sampled_unlab_size', sampled_unlab_size)
        print('unlabeled_size', len(new_unlab_train_ds))


        for i in range(n_split):

            subset = Subset(new_unlab_train_ds, torch.arange(i*sampled_unlab_size, (i+1) * sampled_unlab_size).tolist())

            self.unlab_samp_list.append(DataLoader(subset, batch_size=self.Main_AL_class.batch_size, shuffle=False, num_workers=2))
                            
        return sampled_unlab_size



    def get_A(self):
        
        embeddings_cat = torch.cat((self.labeled_embeddings, self.samp_unlab_embeddings), dim=0).to(self.Main_AL_class.device)

        # Computing Cosine Similarity
        if(self.params['affinity_method'] == 'cosine_similarity'):
            normalized_embedding = F.normalize(embeddings_cat, p=2, dim=-1).to(self.Main_AL_class.device)
            self.A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.Main_AL_class.device)
            del normalized_embedding

        # Calculate Gaussian kernel
        elif(self.params['affinity_method'] == 'gaussian_kernel'):
            self.A = torch.exp(-(torch.cdist(embeddings_cat, embeddings_cat)).pow(2) / (2.0 * 1**2))

        # Calculate the Euclidean Distance
        elif(self.params['affinity_method'] == 'eucliden_distance'):
            self.A = torch.cdist(embeddings_cat, embeddings_cat).to(self.Main_AL_class.device)
        
        else:
            raise ValueError('Invalid Affinity Method, please insert one of the cosine_similarity, gaussian_kernel, or eucliden_distance')




    def get_X(self, len_samp_unlab_embeds):
        
        self.X = torch.zeros(len(self.lab_train_ds) + len_samp_unlab_embeds, self.Main_AL_class.n_classes, dtype=torch.float).to(self.Main_AL_class.device)
        
        for idx, (_, label) in enumerate(self.lab_train_ds): self.X[idx][label] = 1
        for idx in range(len(self.lab_train_ds), len(self.lab_train_ds) + len_samp_unlab_embeds):
            for label in range(self.Main_AL_class.n_classes):
                self.X[idx][label] = 1 / self.Main_AL_class.n_classes
                    



    def gtg(self, tol, max_iter, idx_split, dim_split):
        err = float('Inf')
        i = 0
        
        while err > tol and i < max_iter:
            X_old = self.X.clone()
            self.X = self.X * torch.mm(self.A, self.X)
            self.X /= self.X.sum(axis=1, keepdim=True)
        
            iter_entropy = entropy(self.X) # both labeled and sample unlabeled
            # I have to map only the sample_unlabeled to the correct position
                        
            for idx, unlab_ent_val in enumerate(iter_entropy[len(self.lab_train_ds):]):
                # I iterate only the sampled unlabeled one
                if i == 0:
                    self.entropy_pairwise_der[idx + (idx_split * dim_split)][i] = unlab_ent_val
                else:
                    self.entropy_pairwise_der[idx + (idx_split * dim_split)][i - 1] = unlab_ent_val - self.entropy_pairwise_der[idx + (idx_split * dim_split)][i - 1]
            
                
            err = torch.norm(self.X - X_old)
            i += 1



    def get_new_dataloaders(self, overall_topk):

        new_lab_train_ds = np.array([
            np.array([
                image if isinstance(image, np.ndarray) else image.numpy(), label
            ], dtype=object) for image, label in self.lab_train_ds], dtype=object)

        new_unlab_train_ds = np.array([
            np.array([
                image if isinstance(image, np.ndarray) else image.numpy(), label
            ], dtype=object) for image, label in self.unlab_train_ds], dtype=object)

        
        print('new_lab_train_ds', len(new_lab_train_ds))
        print('new_unlab_train_ds', len(new_unlab_train_ds))
        
        for topk_idx in overall_topk: #tqdm(overall_topk, total=len(overall_topk), leave=False, desc='Adding the observation to the Labeled Dataset'):
            new_lab_train_ds = np.vstack((new_lab_train_ds, np.expand_dims(
                np.array([new_unlab_train_ds[topk_idx.item()][0],
                          new_unlab_train_ds[topk_idx.item()][1]
                ], dtype=object)
            , axis=0)))
            

        for topk_idx in overall_topk: #tqdm(overall_topk, total=len(overall_topk), leave=False, desc='Removing the observation from the Unlabeled Dataset'):
            new_unlab_train_ds[topk_idx.item()] = np.array([np.nan, np.nan], dtype=object)
            # set a [np.nan np.nan] the row and the get all the row not equal to [np.nan, np.nan]
        
        
        print('new_lab_train_ds', len(new_lab_train_ds))
        
        new_unlab_train_ds = new_unlab_train_ds[
            np.array(
                [not np.isnan(row[1])
                    for row in tqdm(new_unlab_train_ds, total=len(new_unlab_train_ds),
                                    leave=False, desc='Obtaining the unmarked observation from the Unlabeled Dataset')
                ]
            )]

        print('new_unlab_train_ds', len(new_unlab_train_ds))
        
        self.lab_train_ds = CIFAR10(None, new_lab_train_ds)
        self.unlab_train_ds = CIFAR10(None, new_unlab_train_ds)

        self.lab_train_dl = DataLoader(self.lab_train_ds, batch_size=self.Main_AL_class.batch_size, shuffle=True)




    def run(self, al_iters, epochs, n_top_k_obs):
        results = {}
        
        for n_splits in self.params['list_n_samples']:
                    
            print(colored(f'----------------------- WORKING WITH {n_splits} UNLABELED SPLITS -----------------------\n', 'green'))
                    
            iter = 0

            results[n_splits] = { 'test_loss': [], 'test_accuracy': [] }
                
            # iter = 0            
            print(colored(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n', 'blue'))
            self.Main_AL_class.reintialize_model()
            self.Main_AL_class.fit(epochs, self.lab_train_dl) # train in the labeled observations
                
            test_loss, test_accuracy = self.Main_AL_class.test_AL()
                
            write_csv(
                ts_dir=self.Main_AL_class.timestamp,
                head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                values = [self.method_name, iter, n_splits, test_accuracy, test_loss]
            )
                
            results[n_splits]['test_loss'].append(test_loss)
            results[n_splits]['test_accuracy'].append(test_accuracy)
                     
                     
            # start of the loop   
            while len(self.unlab_train_ds) > 0 and iter < al_iters:
                print(colored(f'----------------------- ITERATION {iter + 1} / {al_iters} -----------------------\n', 'blue'))            
                
                sampled_unlab_size = self.get_unlabeled_samples(n_splits)
                self.labeled_embeddings = self.Main_AL_class.get_embeddings('Labeled', self.lab_train_dl)
                
                self.entropy_pairwise_der = torch.zeros((len(self.unlab_train_ds), self.params['gtg_max_iter'] - 1))


                pbar = tqdm(enumerate(self.unlab_samp_list), total=len(self.unlab_samp_list), leave=True)
                                
                for idx, unlab_sample_dl in pbar:
                    
                    pbar.set_description(f'WORKING WITH UNLABELED SAMPLE # {idx + 1}')

                    #print(f'----------- WORKING WITH UNLABELED SAMPLE # {idx + 1} -----------\n')

                    self.samp_unlab_embeddings = self.Main_AL_class.get_embeddings('Unlabeled', unlab_sample_dl)
                                
                    self.get_A()
                    self.get_X(len(self.samp_unlab_embeddings))
                    self.gtg(self.params['gtg_tol'], self.params['gtg_max_iter'], idx, sampled_unlab_size)
                    
                    self.clear_memory()

                
                overall_topk = torch.topk(-(torch.sum(self.entropy_pairwise_der, dim = 1) / self.entropy_pairwise_der.shape[1]), n_top_k_obs)           
                            
                self.get_new_dataloaders(overall_topk.indices)
                
                
                # iter + 1
                self.Main_AL_class.reintialize_model()
                self.Main_AL_class.fit(epochs, self.lab_train_dl) # train in the labeled observations
                
                test_loss, test_accuracy = self.Main_AL_class.test_AL()
                
                write_csv(
                    ts_dir=self.Main_AL_class.timestamp,
                    head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                    values = [self.method_name, iter + 1, n_splits, test_accuracy, test_loss]
                )
                
                results[n_splits]['test_loss'].append(test_loss)
                results[n_splits]['test_accuracy'].append(test_accuracy)
                        
                iter += 1
        
        return results