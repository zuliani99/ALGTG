
from utils import get_overall_top_k, write_csv
from cifar10 import CIFAR10

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from tqdm import tqdm
import warnings
import numpy as np
from termcolor import colored



class GTG():
    
    def __init__(self, Main_AL_class, our_methods_params):
        self.method_name = 'AL_GTG'
        self.Main_AL_class = Main_AL_class
        
        self.params = our_methods_params
   
        self.lab_train_dl = self.Main_AL_class.lab_train_dl
        self.unlab_train_dl = self.Main_AL_class.unlab_train_dl
        
        self.lab_train_ds = self.Main_AL_class.lab_train_ds
        self.unlab_train_ds = self.Main_AL_class.unlab_train_ds




    def get_unlabeled_samples(self, n_split):
        self.unlab_samp_list = []

        new_unlab_train_ds = self.unlab_train_ds
        unlabeled_size = len(new_unlab_train_ds)
        sampled_unlab_size = int(unlabeled_size // n_split)
        
        #print('unlabeled_size', unlabeled_size)
        #print('sampled_unlab_size', sampled_unlab_size)

        while (unlabeled_size > 0):

            unlabeled_size -= sampled_unlab_size

            if(unlabeled_size > 0):
                #print('unlabeled_size', unlabeled_size)
                # here I have random sampled from the unalbeled observation, uo
                sampled_unlab_ds, new_unlab_train_ds = random_split(new_unlab_train_ds, [int(sampled_unlab_size), int(unlabeled_size)])

                self.unlab_samp_list.append((sampled_unlab_ds,
                                         DataLoader(sampled_unlab_ds, batch_size=self.Main_AL_class.batch_size, shuffle=False, num_workers=2)))
        
        return sampled_unlab_size


    def get_A(self):
        
        embeddings_cat = torch.cat((self.labeled_embeddings, self.unlabeled_embeddings), dim=0).to(self.Main_AL_class.device)

        # Computing Cosine Similarity
        if(self.params['affinity_method'] == 'cosine_similarity'):
            normalized_embedding = F.normalize(embeddings_cat, p=2, dim=-1).to(self.Main_AL_class.device)
            self.A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.Main_AL_class.device)

        # Calculate Gaussian kernel
        elif(self.params['affinity_method'] == 'gaussian_kernel'):
            self.A = torch.exp(-(torch.cdist(embeddings_cat, embeddings_cat)).pow(2) / (2.0 * 1**2))

        # Calculate the Euclidean Distance
        elif(self.params['affinity_method'] == 'eucliden_distance'):
            self.A = torch.cdist(embeddings_cat, embeddings_cat).to(self.Main_AL_class.device)
        
        else:
            raise ValueError('Invalid Affinity Method, please insert one of the cosine_similarity, gaussian_kernel, or eucliden_distance')




    def get_X(self, len_unlab_embeds):

        self.X = torch.empty(0, self.Main_AL_class.n_classes).to(self.Main_AL_class.device)

        # X for the labeled observations
        for (_, label) in self.lab_train_ds:
            arr_one_zeros = torch.zeros(1, self.Main_AL_class.n_classes).to(self.Main_AL_class.device)
            arr_one_zeros[0][label] = 1
            self.X = torch.cat((self.X, arr_one_zeros), dim=0)

        self.X = torch.cat((self.X, torch.full((len_unlab_embeds, self.Main_AL_class.n_classes), 1/self.Main_AL_class.n_classes).to(self.Main_AL_class.device)), dim=0)




    def gtg(self, tol, max_iter):
        err = float('Inf')
        i = 0

        while err > tol and i < max_iter:
            X_old = self.X.clone()
            self.X = self.X * torch.mm(self.A, self.X)
            self.X /= self.X.sum(axis=1, keepdim=True)

            err = torch.norm(self.X - X_old)
            i += 1

        if i == max_iter:
            warnings.warn('Max number of iterations reached.')

        return i



    def entropy_topK(self, top_k):
        return torch.topk(-torch.sum(self.X * torch.log2(self.X + 1e-20), dim=1), top_k)



    def get_new_dataloaders(self, overall_topk):

        new_lab_train_ds = np.array([
            np.array([
                image if isinstance(image, np.ndarray) else image.numpy(), label
            ], dtype=object) for image, label in tqdm(self.lab_train_ds, total=len(self.lab_train_ds), leave=False, desc='Copying lab_train_ds')], dtype=object)
        
        #new_lab_train_ds = torch.tensor([self.lab_train_ds[i][::1]] for i in tqdm(range(len(self.lab_train_ds))))
        
        new_unlab_train_ds = np.array([
            np.array([
                image if isinstance(image, np.ndarray) else image.numpy(), label
            ], dtype=object) for image, label in tqdm(self.unlab_train_ds, total=len(self.unlab_train_ds), leave=False, desc='Copying unlab_train_ds')], dtype=object)
        
        
        #print('self.lab_train_ds', len(self.lab_train_ds), 'new_lab_train_ds', len(new_lab_train_ds))
        #print('self.unlab_train_ds', len(self.unlab_train_ds), 'new_unlab_train_ds', len(new_unlab_train_ds))
        
        
        #print(overall_topk, len(overall_topk))
        
        #new_unlab_train_ds = torch.tensor([self.unlab_train_ds[i][::1]] for i in tqdm(range(len(self.unlab_train_ds))))

        #for list_index, topk_index_value in tqdm(overall_topk, total=len(overall_topk), leave=False, desc='Adding the observation to the Labeled Dataset'):
        for topk_idx in tqdm(overall_topk, total=len(overall_topk), leave=False, desc='Adding the observation to the Labeled Dataset'):
            new_lab_train_ds = np.vstack((new_lab_train_ds, np.expand_dims(
                np.array([new_unlab_train_ds[topk_idx.item()][0],
                          new_unlab_train_ds[topk_idx.item()][1]
                ], dtype=object)
            , axis=0)))
            
            #new_lab_train_ds = torch.cat((new_lab_train_ds, torch.tensor([self.unlab_samp_list[list_index][0][topk_index_value][::1]])), dim = 0)


        
        #to_nan = []
        
        #for list_index, topk_index_value in tqdm(overall_topk, total=len(overall_topk), leave=False, desc='Removing the observation from the Unlabeled Dataset'):
        for topk_idx in tqdm(overall_topk, total=len(overall_topk), leave=False, desc='Removing the observation from the Unlabeled Dataset'):
            
            #print((list_index * n_samples) + topk_index_value)
            
            #if((list_index * n_samples) + topk_index_value in to_nan): 
            #    raise Exception('idice già vistoooo!')
            
            #if np.isnan(new_unlab_train_ds[(list_index * n_samples) + topk_index_value][1]):
            #if np.isnan(new_unlab_train_ds[topk_idx.item()][1]):
            #    raise Exception('GIAAAAAAA NAN')
            
            #new_unlab_train_ds[(list_index * n_samples) + topk_index_value] = np.array([np.nan, np.nan], dtype=object) # set a [np.nan np.nan] the row and the get all the row not equal to [np.nan, np.nan]
            new_unlab_train_ds[topk_idx.item()] = np.array([np.nan, np.nan], dtype=object) # set a [np.nan np.nan] the row and the get all the row not equal to [np.nan, np.nan]
            
            #to_nan.append((list_index * n_samples) + topk_index_value)
            
            
            #new_unlab_train_ds = torch.cat((new_unlab_train_ds[ : ((list_index * n_samples) + topk_index_value)],
            #                                new_unlab_train_ds[((list_index * n_samples) + topk_index_value + 1) : ]))
            
        #i = 0
        #for row in new_unlab_train_ds: 
        #    if(np.isnan(row[1])): i += 1
            
        #print(f'ci sono {i} ossevazioni maskerate e da cancellare')
        
        
        
        
        print('new_lab_train_ds', len(new_lab_train_ds))
        
        new_unlab_train_ds = new_unlab_train_ds[np.array([not np.isnan(row[1]) for row in tqdm(new_unlab_train_ds, total=len(new_unlab_train_ds), leave=False, desc='Obtaining the unmarked observation from the Unlabeled Dataset')])]

        print('new_unlab_train_ds', len(new_unlab_train_ds))
        
        self.lab_train_ds = CIFAR10(None, new_lab_train_ds)
        self.unlab_train_ds = CIFAR10(None, new_unlab_train_ds)

        self.lab_train_dl = DataLoader(self.lab_train_ds, batch_size=self.Main_AL_class.batch_size, shuffle=True)
        self.unlab_train_dl = DataLoader(self.unlab_train_ds, batch_size=self.Main_AL_class.batch_size, shuffle=True)




    def run(self, al_iters, epochs, n_top_k_obs):
        results = {}
        
        for n_splits in self.params['list_n_samples']:
                    
            print(colored(f'----------------------- WORKING WITH {n_splits} UNLABELED SPLITS -----------------------\n', 'green'))
                    
            iter = 0

            results[n_splits] = { 'test_loss': [], 'test_accuracy': [] }
                
            # iter = 0            
            print(colored(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n', 'blue'))
            #self.Main_AL_class.reintialize_model()
            self.Main_AL_class.fit(epochs, self.lab_train_dl) # train in the labeled observations
                
            test_loss, test_accuracy = self.Main_AL_class.test_AL()
                
            write_csv(
                #filename = 'OUR_test_res.csv',
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

                ds_top_k = []

                pbar = tqdm(enumerate(self.unlab_samp_list), total=len(self.unlab_samp_list), leave=True)
                                
                for idx, (_, unlab_sample_dl) in pbar:#enumerate(self.unlab_samp_list):
                    
                    pbar.set_description(f'WORKING WITH UNLABELED SAMPLE # {idx + 1}')

                    #print(f'----------- WORKING WITH UNLABELED SAMPLE # {idx + 1} -----------\n')

                    self.unlabeled_embeddings = self.Main_AL_class.get_embeddings('Unlabeled', unlab_sample_dl)
                                
                    self.get_A()
                    self.get_X(len(self.unlabeled_embeddings))
                    self.gtg(self.params['gtg_tol'], self.params['gtg_max_iter'])

                    topk_idx_val_obs = self.entropy_topK(n_top_k_obs) # top k for the matrix X composed with the ds of labeled and unlabeled, so the index are referred to these two sets

                    ds_top_k.append(topk_idx_val_obs)

            
                overall_topk = get_overall_top_k(ds_top_k, n_top_k_obs, sampled_unlab_size, self.Main_AL_class.device)
                #tensor of indices of the unlabeled dataset
                            
                self.get_new_dataloaders(overall_topk)
                
                
                # iter + 1
                self.Main_AL_class.reintialize_model()
                self.Main_AL_class.fit(epochs, self.lab_train_dl) # train in the labeled observations
                
                test_loss, test_accuracy = self.Main_AL_class.test_AL()
                
                write_csv(
                    #filename = 'OUR_test_res.csv',
                    ts_dir=self.Main_AL_class.timestamp,
                    head = ['method', 'al_iter', 'n_splits', 'test_accuracy', 'test_loss'],
                    values = [self.method_name, iter + 1, n_splits, test_accuracy, test_loss]
                )
                
                results[n_splits]['test_loss'].append(test_loss)
                results[n_splits]['test_accuracy'].append(test_accuracy)
                
                
                        
                iter += 1
        
        return results