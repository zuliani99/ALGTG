
import torch
from torch.utils.data import DataLoader, Subset

from torchvision import transforms
import copy
from CIFAR10 import CIFAR10, Cifar10SubsetDataloaders
from utils import get_mean_std

import os


class TrainEvaluate(object):

    def __init__(self, params):
        
        self.best_check_filename = 'app/checkpoints'
        self.init_check_filename = 'app/checkpoints/init_checkpoint.pth.tar'
        
        cifar10: Cifar10SubsetDataloaders = params['cifar10']
        
        self.n_classes = len(cifar10.classes)
        self.device = params['device']
        self.batch_size = params['batch_size']
        self.score_fn = params['score_fn']
        self.patience = params['patience']
        self.timestamp = params['timestamp']
        self.loss_fn = params['loss_fn']
        
        self.lab_train_dl: DataLoader = copy.deepcopy(cifar10.lab_train_dl)
        self.lab_train_subset: Subset = copy.deepcopy(cifar10.lab_train_subset)
        self.unlab_train_subset: Subset = copy.deepcopy(cifar10.unlab_train_subset)
        
        self.test_dl: DataLoader = cifar10.test_dl
        self.val_dl: DataLoader = cifar10.val_dl
        self.original_trainset: CIFAR10 = cifar10.original_trainset

        self.model = params['model'].to(self.device)
        self.optimizer = params['optimizer']
        self.scheduler = params['scheduler']
        self.loss_weird = params['loss_weird']
            
        self.obtain_normalization()
        
        
        

    def reintialize_model(self):
        print(' => Load Initial Checkpoint')
        self.__load_init_checkpoint()
        print(' DONE\n')



    def obtain_normalization(self):
        mean, std = get_mean_std(self.lab_train_dl)
        self.normalize = transforms.Compose([ transforms.Normalize(mean, std) ])  
        
        
        
    def __save_best_checkpoint(self, filename):
        checkpoint = {
            'state_dict': self.model.state_dict(), 
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, filename)

    
    
    def __load_init_checkpoint(self):

        checkpoint = torch.load(self.init_check_filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        


    def __load_best_checkpoint(self, filename):

        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        


    def evaluate(self, dataloader, flag_loss_weird = False):
        tot_accuracy, tot_loss = .0, .0

        self.model.eval()

        with torch.inference_mode(): # Allow inference mode
            for _, images, labels in dataloader:
                
                images, labels = self.normalize(images.to(self.device)), labels.to(self.device)

                outputs, _, out_weird, _ = self.model(images)

                loss_ce = self.loss_fn(outputs, labels)
                
                if flag_loss_weird:    
                    loss_weird = self.loss_weird(out_weird, loss_ce)
                    loss_ce = torch.mean(loss_ce)
                    loss = loss_ce + loss_weird
                else:
                    loss = torch.mean(loss_ce)
                
                accuracy = self.score_fn(outputs, labels)

                tot_accuracy += accuracy
                tot_loss += loss.item()


            tot_accuracy /= len(dataloader)
            tot_loss /= len(dataloader)
            
        return tot_accuracy, tot_loss




    def train_evaluate(self, epochs, dataloader, method_str):
        
        #resnet_weird
        weight = 1.   # 120 = 0
    
        check_best_path = f'{self.best_check_filename}/best_{method_str}.pth.tar'
		
        best_val_loss = float('inf')
        actual_patience = 0
        
        
        results = { 'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [] }
	

        for epoch in range(epochs):
            
            self.model.train()
            
            #resnet_weird
            if epoch > 70: weight = 0
            
            train_loss = 0.0
            train_accuracy = 0.0
            

            for _, images, labels in dataloader:

                # get the inputs; data is a list of [inputs, labels]
                images, labels = self.normalize(images.to(self.device)), labels.to(self.device)
                
                
                self.optimizer.zero_grad()
                                
                # learning loss
                outputs, _, out_weird, _ = self.model(images)
                
                loss_ce = self.loss_fn(outputs, labels)
                    
                loss_weird = self.loss_weird(out_weird, loss_ce)
                loss_ce = torch.mean(loss_ce)
                    
                loss = loss_ce + weight * loss_weird
                    
                
                #outputs, _, _, _ = self.model(images)
                #loss = self.loss_fn(outputs, labels)
                
                
                loss.backward()         
                self.optimizer.step()
                
                accuracy = self.score_fn(outputs, labels)
                
                train_loss += loss.item()
                train_accuracy += accuracy

    

            train_accuracy /= len(dataloader)
            train_loss /= len(dataloader)
            
            
            val_accuracy, val_loss = self.evaluate(self.val_dl, weight)
            
            
            # CosineAnnealingLR
            #self.scheduler.step()
            #print(self.scheduler.get_last_lr())
            

            results['train_loss'].append(train_loss)
            results['train_accuracy'].append(train_accuracy)
            results['val_loss'].append(val_loss)
            results['val_accuracy'].append(val_accuracy)
            
            
            #print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, val_loss: {:.6f}, best_val_loss: {:.6f} \n'.format(
            #    epoch + 1, train_accuracy, train_loss, val_accuracy, val_loss, best_val_loss))
            #print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, val_loss: {:.6f}, best_val_loss: {:.6f}\n'.format(
            #          epoch + 1, train_accuracy, train_loss, val_accuracy, val_loss, val_loss if val_loss < best_val_loss else best_val_loss))


            if(val_loss < best_val_loss):
                best_val_loss = val_loss
                actual_patience = 0
                print(' => Saving best checkpoint')
                self.__save_best_checkpoint(check_best_path)
                print(' DONE\n')
            else:
                actual_patience += 1
                if actual_patience >= self.patience:
                    
                    print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, val_loss: {:.6f}, best_val_loss: {:.6f} \n'.format(
                        epoch + 1, train_accuracy, train_loss, val_accuracy, val_loss, best_val_loss))
                    
                    print(f'Early stopping, validation loss do not decreased for {self.patience} epochs')
                    break
                
                
            print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, val_loss: {:.6f}, best_val_loss: {:.6f} \n'.format(
                epoch + 1, train_accuracy, train_loss, val_accuracy, val_loss, best_val_loss))

            if epoch == 100:
                print('Decreasing learning rate to 0.01 and ignoring the learning loss\n')
                for g in self.optimizer.param_groups: g['lr'] = 0.01
                    

        self.__load_best_checkpoint(check_best_path)

        print('Finished Training\n')
        
        return {'model_name': method_str, 'results': results}



    def test(self):
        test_accuracy, test_loss = self.evaluate(self.test_dl)

        print('TESTING RESULTS -> test_accuracy: {:.6f}, test_loss: {:.6f} \n\n'.format(test_accuracy, test_loss))

        return test_accuracy, test_loss



    def get_embeddings(self, dataloader):

        embeddings = torch.empty(0, self.model.linear.in_features, dtype=torch.float32).to(self.device)
        concat_labels = torch.empty(0, dtype=torch.int8).to(self.device)
        self.model.eval()

        # again no gradients needed
        with torch.inference_mode():
            for _, images, labels in dataloader:
                
                images = self.normalize(images.to(self.device))
                labels = labels.to(self.device)
                
                _, embed, _, _ = self.model(images)
                
                embeddings = torch.cat((embeddings, embed.squeeze()), dim=0)
                concat_labels = torch.cat((concat_labels, labels), dim=0)
             
        return embeddings, concat_labels



    def get_new_dataloaders(self, overall_topk):
        
        # temp variable
        lab_train_indices = self.lab_train_subset.indices
        
        # extend with the overall_topk
        lab_train_indices.extend(overall_topk)
        # generate a new Subset
        self.lab_train_subset = Subset(self.original_trainset, lab_train_indices)
        
        
        # update the indices for the transform        
        self.original_trainset.lab_train_idxs = lab_train_indices
        
        # temp variable
        unlab_train_indices = self.unlab_train_subset.indices
        # remove new labeled observations
        for idx_to_remove in overall_topk: unlab_train_indices.remove(idx_to_remove)
        # generate a new Subset
        self.unlab_train_subset = Subset(self.original_trainset, unlab_train_indices)
        
        # sanity check
        if len(list(set(self.unlab_train_subset.indices) & set(self.lab_train_subset.indices))) == 0:
            print('Intersection between indices is EMPTY')
        else: raise Exception('NON EMPTY INDICES INTERSECTION')

        # generate the new labeled DataLoader
        self.lab_train_dl = DataLoader(self.lab_train_subset, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        self.obtain_normalization()
