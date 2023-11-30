from cifar10 import CIFAR10
from utils import get_overall_top_k

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

#from tqdm.notebook import tqdm
from tqdm import tqdm


class GTG_ActiveLearning():


    def __init__(self, n_classes, batch_size, model, optimizer, train_dl, test_dl, splitted_train_dl, splitted_train_ds, loss_fn, val_dl, score_fn, scheduler, device, patience, affinity_method):
        self.n_classes = n_classes
        self.model = model.to(device)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.lab_train_dl, self.unlab_train_dl = splitted_train_dl
        self.lab_train_ds, self.unlab_train_ds = splitted_train_ds
        self.loss_fn = loss_fn
        self.val_dl = val_dl
        self.score_fn = score_fn
        self.scheduler = scheduler
        self.device = device
        self.affinity_method = affinity_method
        self.patience = patience
        self.best_check_filename = 'app/checkpoints/best_checkpoint.pth.tar'
        self.init_check_filename = 'app/checkpoints/init_checkpoint.pth.tar'
        self.__save_checkpoint(self.init_check_filename)

        self.labeled_embeddings = torch.empty(0, list(self.model.children())[-1].in_features).to(self.device)
        self.unlabeled_embeddings = torch.empty(0, list(self.model.children())[-1].in_features).to(self.device)

        self.embed_model = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        
        

    def __reintialize_model(self):
        self.__load_checkpoint(self.init_check_filename, 'Initial')



    def __save_checkpoint(self, filename):

        print(f'=> Saving Checkpoint to {filename}')
        checkpoint = { 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict() }
        torch.save(checkpoint, filename)
        print(' DONE\n')



    def __load_checkpoint(self, filename, type_load):

        print(f'=> Loading {type_load} Checkpoint')
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(' DONE\n')



    def evaluate(self, val_dl, epoch = 0, epochs = 0):
        val_loss, val_accuracy = .0, .0

        self.model.eval()

        pbar = tqdm(enumerate(val_dl), total = len(val_dl), leave=False)

        with torch.inference_mode(): # Allow inference mode
            for _, (images, label) in pbar:
                images, label = images.to(self.device), label.to(self.device)

                output = self.model(images)

                loss = self.loss_fn(output, label)

                accuracy = self.score_fn(output, label)

                val_loss += loss.item()
                val_accuracy += accuracy

                if epoch > 0: pbar.set_description(f'EVALUATION Epoch [{epoch} / {epochs}]')
                else: pbar.set_description(f'TESTING')
                pbar.set_postfix(loss = loss.item(), accuracy = accuracy)

            val_loss /= len(val_dl) # Calculate the final loss
            val_accuracy /= len(val_dl)
        return val_loss, val_accuracy



    def fit(self, epochs):
        self.model.train()

        best_val_loss = float('inf')
        actual_patience = 0

        for epoch in range(epochs):  # loop over the dataset multiple times

            train_loss = 0.0
            train_accuracy = 0.0

            pbar = tqdm(enumerate(self.lab_train_dl), total = len(self.lab_train_dl), leave=False)

            for _, (inputs, labels) in pbar:
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                loss.backward()
                self.optimizer.step()

                accuracy = self.score_fn(outputs, labels)

                # print statistics
                train_loss += loss.item()
                train_accuracy += accuracy

                # Update the progress bar
                pbar.set_description(f'TRAIN Epoch [{epoch + 1} / {epochs}]')
                pbar.set_postfix(loss = loss.item(), accuracy = accuracy)

            train_loss /= len(self.lab_train_dl)
            train_accuracy /= len(self.lab_train_dl)

            self.scheduler.step(train_loss)

            # Validation step
            val_loss, val_accuracy = self.evaluate(self.val_dl, epoch + 1, epochs)

            print('Epoch [{}], train_loss: {:.6f}, train_accuracy: {:.6f}, val_loss: {:.6f}, val_accuracy: {:.6f} \n'.format(
                      epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy))

            if(val_loss < best_val_loss):
                best_val_loss = val_loss
                actual_patience = 0
                self.__save_checkpoint(self.best_check_filename)
            else:
                actual_patience += 1
                if actual_patience >= self.patience:
                    print(f'Early stopping, validation loss do not decreased for {self.patience} epochs')
                    pbar.close() # Closing the progress bar before exiting from the train loop
                    break

        self.__load_checkpoint(self.best_check_filename, 'Best')

        print('Finished Training\n')



    def get_embeddings(self, mode_type, dataloader):

        print(f'{mode_type} Embedding Computation')

        if mode_type == 'Sampled':
            self.unlabeled_embeddings = torch.empty(0, list(self.model.children())[-1].in_features).to(self.device)
        else:
            self.labeled_embeddings = torch.empty(0, list(self.model.children())[-1].in_features).to(self.device)

        self.embed_model.eval()

        pbar = tqdm(enumerate(dataloader), total = len(dataloader), leave=False)

        # again no gradients needed
        with torch.inference_mode():
            for _, (inputs, _) in pbar:
                pbar.set_description(f'Getting {mode_type} Embeddings')
                embed = self.embed_model(inputs.to(self.device))

                if mode_type == 'Labeled':
                    self.labeled_embeddings = torch.cat((self.labeled_embeddings, embed.squeeze()), dim=0)
                else:
                    self.unlabeled_embeddings = torch.cat((self.unlabeled_embeddings, embed.squeeze()), dim=0)

        print(' => DONE\n')



    def get_unlabeled_samples(self, n_split):
        self.unlab_samp_list = []

        new_unlab_train_ds = self.unlab_train_ds
        unlabeled_size = len(new_unlab_train_ds)
        sampled_unlab_size = int(unlabeled_size // n_split)

        while (unlabeled_size > 0):

            unlabeled_size -= sampled_unlab_size

            # here I have random sampled from the unalbeled observation, uo
            sampled_unlab_ds, new_unlab_train_ds = random_split(new_unlab_train_ds, [int(sampled_unlab_size), int(unlabeled_size)])

            self.unlab_samp_list.append((sampled_unlab_ds,
                                         DataLoader(sampled_unlab_ds, batch_size=self.batch_size, shuffle=False, num_workers=2)))




    def get_A(self):
        
        print('Obtaining Affinity Matrix')
        print(self.labeled_embeddings.shape, self.unlabeled_embeddings.shape)
        
        embeddings_cat = torch.cat((self.labeled_embeddings, self.unlabeled_embeddings), dim=0).to(self.device)

        # Computing Cosine Similarity
        if(self.affinity_method == 'cosine_similarity'):
            normalized_embedding = F.normalize(embeddings_cat, p=2, dim=-1).to(self.device)
            self.A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device)

        # Calculate Gaussian kernel
        elif(self.affinity_method == 'gaussian_kernel'):
            self.A = torch.exp(-(torch.cdist(embeddings_cat, embeddings_cat)).pow(2) / (2.0 * 1**2))

        # Calculate the Euclidean Distance
        elif(self.affinity_method == 'eucliden_distance'):
            self.A = torch.cdist(embeddings_cat, embeddings_cat).to(self.device)
        
        else:
            raise ValueError('Invalid Affinity Method, please insert one of the cosine_similarity, gaussian_kernel, or eucliden_distance')
            
        print(self.A.shape)
        print(f' => DONE with memory usage {self.A.element_size() * self.A.nelement()} bytes\n')



    def get_X(self):
        print('Obtaining Initial X Matrix')

        self.X = torch.empty(0, self.n_classes).to(self.device)

        # X for the labeled observations
        for (_, label) in self.lab_train_ds:
            arr_one_zeros = torch.zeros(1, self.n_classes).to(self.device)
            arr_one_zeros[0][label] = 1
            self.X = torch.cat((self.X, arr_one_zeros), dim=0)

        self.X = torch.cat((self.X, torch.full((len(self.unlabeled_embeddings), self.n_classes), 1/self.n_classes).to(self.device)), dim=0)
        print(self.X.shape)
        print(f' => DONE with memory usage {self.X.element_size() * self.X.nelement()} bytes\n')



    def gtg(self, tol, max_iter):
        err = float('Inf')
        i = 0

        print('Runnning GTG Algorithm')

        while err > tol and i < max_iter:
            X_old = self.X.clone()
            self.X = self.X * torch.mm(self.A, self.X)
            self.X /= self.X.sum(axis=1, keepdim=True)

            err = torch.norm(self.X - X_old)
            i += 1

        if i == max_iter:
            warnings.warn('Max number of iterations reached.')

        print(f' => DONE with {i} iterations\n')

        return i



    def get_topK_obs(self, top_k):
        print('Obtaining the top_k most interesting observations')

        self.topk_idx_val_obs = torch.topk(-torch.sum(self.X * torch.log2(self.X + 1e-20), dim=1), top_k)

        print(' => DONE\n')



    def get_new_dataloader(self, overall_topk, n_samples):

        print('Copying the labeled train dataset')
        new_lab_train_ds = np.array([
            np.array([
                self.lab_train_ds[i][0] if isinstance(self.lab_train_ds[i][0], np.ndarray) else self.lab_train_ds[i][0].numpy(),
                self.lab_train_ds[i][1]
            ], dtype=object) for i in tqdm(range(len(self.lab_train_ds)))], dtype=object)
        
        #new_lab_train_ds = torch.tensor([self.lab_train_ds[i][::1]] for i in tqdm(range(len(self.lab_train_ds))))
        
        print(' => DONE\n')

        print('Copying the unlabeled train dataset')
        new_unlab_train_ds = np.array([
            np.array([
                self.unlab_train_ds[i][0] if isinstance(self.unlab_train_ds[i][0], np.ndarray) else self.unlab_train_ds[i][0].numpy(),
                self.unlab_train_ds[i][1]
            ], dtype=object) for i in tqdm(range(len(self.unlab_train_ds)))], dtype=object)
        
        #new_unlab_train_ds = torch.tensor([self.unlab_train_ds[i][::1]] for i in tqdm(range(len(self.unlab_train_ds))))
        
        print(' => DONE\n')


        l = len(new_lab_train_ds)

        print(f'Expanding the labeled train dataset {len(new_lab_train_ds)}')
        for list_index, topk_index_value in overall_topk:
            new_lab_train_ds = np.vstack((new_lab_train_ds, np.expand_dims(
                np.array([
                    self.unlab_samp_list[list_index][0][topk_index_value][0] if isinstance(self.unlab_samp_list[list_index][0][topk_index_value][0], np.ndarray)
                        else self.unlab_samp_list[list_index][0][topk_index_value][0].numpy(),
                    self.unlab_samp_list[list_index][0][topk_index_value][1]
                ], dtype=object)
            , axis=0)))
            
            #new_lab_train_ds = torch.cat((new_lab_train_ds, torch.tensor([self.unlab_samp_list[list_index][0][topk_index_value][::1]])), dim = 0)
            
        print(new_lab_train_ds.shape)
        print(f' => DONE {len(new_lab_train_ds)}\n')

        print(f'Reducing the unlabeled train dataset {len(new_unlab_train_ds)}')
        for list_index, topk_index_value in overall_topk:
            new_unlab_train_ds = np.delete(new_unlab_train_ds, (list_index * n_samples) + topk_index_value, axis = 0)
            
            #new_unlab_train_ds = torch.cat((new_unlab_train_ds[ : ((list_index * n_samples) + topk_index_value)],
            #                                new_unlab_train_ds[((list_index * n_samples) + topk_index_value + 1) : ]))
            
        print(new_unlab_train_ds.shape)
        print(f' => DONE {len(new_unlab_train_ds)}\n')


        self.lab_train_ds = CIFAR10(None, new_lab_train_ds)
        self.unlab_train_ds = CIFAR10(None, new_unlab_train_ds)

        self.lab_train_dl = DataLoader(self.lab_train_ds, batch_size=self.batch_size, shuffle=True)
        self.unlab_train_dl = DataLoader(self.unlab_train_ds, batch_size=self.batch_size, shuffle=True)



    def test_AL_GTG(self):
        test_loss, test_accuracy = self.evaluate(self.test_dl)

        print('\nTESTING RESULTS -> val_loss: {:.6f}, val_accuracy: {:.6f} \n'.format(test_loss, test_accuracy))

        return test_loss, test_accuracy



    def train_evaluate_AL_GTG(self, epochs, al_iters, gtg_tol, gtg_max_iter, top_k_obs, list_n_samples):

        results = { }
        n_lab_obs =  [len(self.lab_train_ds) + (iter * top_k_obs) for iter in range(al_iters)]
        
        print(f'----------------------- START TRAINING ACTIVE LEARNING -----------------------\n')
        

        for n_samples in list_n_samples:
            
            print(f'----------------------- START {n_samples} unalbeled samples of equal dimension  -----------------------\n')
            
            iter = 0

            results[n_samples] = { 'test_loss': [], 'test_accuracy': [] }

            while iter < al_iters:
                print(f'----------------------- START ITERATION {iter + 1} / {al_iters} -----------------------\n')
                self.__reintialize_model()
                self.fit(epochs) # train in the labeled observations
                self.get_unlabeled_samples(n_samples)
                self.get_embeddings('Labeled', self.lab_train_dl)

                ds_top_k = []

                for idx, (_, unlab_sample_dl) in enumerate(self.unlab_samp_list):

                    print(f'----------- STARTING WORKING WITH UNLABELED SAMPLE # {idx + 1} -----------\n')

                    self.get_embeddings('Sampled', unlab_sample_dl)

                    self.get_A()
                    self.get_X()
                    self.gtg(gtg_tol, gtg_max_iter)

                    self.get_topK_obs(top_k_obs) # top k for the matrix X composed with the ds of labeled and unlabeled, so the index are referred to these two sets

                    ds_top_k.append(self.topk_idx_val_obs)

                    print(f'----------- ENDING WORKING WITH UNLABELED SAMPLE # {idx + 1} -----------\n')

                overall_topk = get_overall_top_k(ds_top_k, top_k_obs)

                self.get_new_dataloader(overall_topk, n_samples)

                test_loss, test_accuracy = self.test_AL_GTG()

                results[n_samples]['test_loss'].append(test_loss)
                results[n_samples]['test_accuracy'].append(test_accuracy)
                
                #n_lab_obs.append(len(self.lab_train_ds))
                
                #print(results, n_lab_obs)

                print(f'----------------------- END ITERATION {iter + 1} / {al_iters} -----------------------\n')
                iter += 1
                
            print(f'----------------------- START {n_samples} unalbeled samples of equal dimension  -----------------------\n')

        return results, n_lab_obs