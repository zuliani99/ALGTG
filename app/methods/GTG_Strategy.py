
from TrainEvaluate import TrainEvaluate
from utils import entropy
from CIFAR10 import UniqueShuffle

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

import copy



class GTG_Strategy(TrainEvaluate):
    
    def __init__(self, al_params, our_methods_params, LL):
        super().__init__(al_params, LL)
                
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.params = our_methods_params 
        self.LL = LL
        
        self.weights = torch.flip(torch.linspace(1, 0.1, self.params['gtg_max_iter'] - 1), [0]).to(self.device)


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
        
        self.X = torch.zeros(
            len(self.lab_train_subset) + len_samp_unlab_embeds,
            self.n_classes,
            dtype=torch.float32,
            device=self.device
        )

        for idx, label in enumerate(target_lab_obs): self.X[idx][label] = 1
        
        for idx in range(len(self.lab_train_subset), len(self.lab_train_subset) + len_samp_unlab_embeds):
            for label in range(self.n_classes):
                self.X[idx][label] = 1. / self.n_classes
        
        

    def gtg(self, indices):
        err = float('Inf')
        i = 0
        
        while err > self.params['gtg_tol'] and i < self.params['gtg_max_iter']:
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
            
    
    
    def get_embeddings(self, dataloader):

        embeddings = torch.empty(0, self.model.linear.in_features, dtype=torch.float32, device=self.device)
        concat_labels = torch.empty(0, dtype=torch.int8, device=self.device)
        self.model.eval()

        # again no gradients needed
        with torch.inference_mode():
            for _, images, labels in dataloader:
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                _, embed, _, _ = self.model(images)
                
                embeddings = torch.cat((embeddings, embed.squeeze()), dim=0)
                concat_labels = torch.cat((concat_labels, labels), dim=0)
             
        return embeddings, concat_labels



    def run(self, al_iters, epochs, unlab_sample_dim, n_top_k_obs):

        iter = 1            

        results = { 'test_accuracy': [], 'test_loss': [] , 'test_loss_ce': [], 'test_loss_weird': []}
                
        print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
        
        # iter = 1
        self.train_evaluate_save(epochs, n_top_k_obs, iter, results)         
                     
        # start of the loop   
        while len(self.unlab_train_subset) > 0 and iter < al_iters:
            iter += 1
                
            print(f'----------------------- ITERATION {iter} / {al_iters} -----------------------\n')
                                                        
            sample_unlab_subset = Subset(
                self.non_transformed_trainset,
                self.get_unlabebled_samples(unlab_sample_dim, iter)
            )
                        
            self.unlab_train_dl = DataLoader(
                sample_unlab_subset,
                batch_size=self.batch_size,
                sampler=UniqueShuffle(sample_unlab_subset),
                pin_memory=True
            )
                
            print(' => Getting the labeled and unlabeled embeddings')
            self.labeled_embeddings, target_lab_obs = self.get_embeddings(self.lab_train_dl)
            self.unlab_embeddings, _ = self.get_embeddings(self.unlab_train_dl)
            print(' DONE\n')
                
            # at each AL round I reinitialize the entropy_pairwise_der since I have to decide at each step what observations I want to move
            self.entropy_pairwise_der = torch.zeros(
                (len(self.transformed_trainset), self.params['gtg_max_iter'] - 1),
                device=self.device
            )

            # for each split

            # split_idx -> indices of the split
            # indices -> are the list of indices for the given batch which ARE CONSISTENT SINCE ARE REFERRED TO THE INDEX OF THE ORIGINAL DATASET
            for split_idx, (indices, _, _) in enumerate(self.unlab_train_dl):
                print(f' => Running GTG for split {split_idx}')
                self.get_A(self.unlab_embeddings[split_idx * self.batch_size : (split_idx + 1) * self.batch_size])
                self.get_X(target_lab_obs, indices.shape[0])
                self.gtg(indices)
                
                self.clear_memory()
                print(' DONE\n')
                                   
                    

            # mean of the entropy derivate 
            #print(torch.sum(self.entropy_pairwise_der, dim = 1, dtype=torch.float32))
            #overall_topk = torch.topk((torch.sum(self.entropy_pairwise_der, dim = 1, dtype=torch.float32) / self.entropy_pairwise_der.shape[1]), n_top_k_obs)
                
                
            # weighted average
            #overall_topk = torch.topk(torch.mean(self.entropy_pairwise_der * self.weights, dim = 1), n_top_k_obs)
                
            # mean only
            overall_topk = torch.topk(torch.mean(self.entropy_pairwise_der, dim = 1), n_top_k_obs)
                                
            #overall_topk.indices -> referred to the matrix indices of entropy_pairwise_der, which are referred to the original trainset
                
            print(' => Modifing the Subsets and Dataloader')
            self.get_new_dataloaders(overall_topk.indices.tolist())
            print(' DONE\n')
                
            # iter + 1
            self.train_evaluate_save(epochs, iter * n_top_k_obs, iter, results)
                                            
        return results