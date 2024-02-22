
import torch
from torch.utils.data import DataLoader, Subset

from ResNet18 import BasicBlock, ResNet_Weird, LearningLoss

import copy
import random
import os

from Datasets import DatasetChoice, SubsetDataloaders
from utils import init_weights_apply, save_train_val_curves, write_csv, create_directory


class TrainEvaluate(object):

    def __init__(self, params, LL):

        self.LL = LL        
        sdl: SubsetDataloaders = params['DatasetChoice']
        
        self.n_classes = sdl.n_classes
        self.n_channels = sdl.n_channels
        #self.image_size = sdl.image_size
        self.device = params['device']
        self.batch_size = params['batch_size']
        self.score_fn = params['score_fn']
        self.patience = params['patience']
        self.timestamp = params['timestamp']
        self.dataset_name = params['dataset_name']
                
        self.best_check_filename = f'app/checkpoints/{self.dataset_name}'
        self.init_check_filename = f'{self.best_check_filename}_init_checkpoint.pth.tar'
        
        # I need the deep copy only of the subsets, that have the indices referred to the original_trainset
        self.lab_train_subset: Subset = copy.deepcopy(sdl.lab_train_subset)
        self.unlab_train_subset: Subset = copy.deepcopy(sdl.unlab_train_subset)
        
        # parameters that are used for all the strategies
        self.lab_train_dl = DataLoader(
            self.lab_train_subset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True
        )
        self.len_lab_train_dl = len(self.lab_train_dl)
        self.test_dl: DataLoader = sdl.test_dl
        self.val_dl: DataLoader = sdl.val_dl
        
        self.transformed_trainset: DatasetChoice = sdl.transformed_trainset 
        self.non_transformed_trainset: DatasetChoice = sdl.non_transformed_trainset 
        
        self.model = ResNet_Weird(BasicBlock, [2, 2, 2, 2], num_classes=self.n_classes, n_channels=self.n_channels).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.loss_weird = LearningLoss(self.device)
        
        if not os.path.exists(self.init_check_filename):
            print(' => Initializing weights')
            self.model.apply(init_weights_apply)
            print(' DONE\n')
            
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        create_directory(self.best_check_filename)

        if not os.path.exists(self.init_check_filename):
            self.__save_checkpoint(self.init_check_filename, 'initial')

        
    def __save_checkpoint(self, filename, check_type):
        print(f' => Saving {check_type} checkpoint')
        checkpoint = {
            'state_dict': self.model.state_dict(), 
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, filename)
        print(' DONE\n')
    
    
    def __load_checkpoint(self, filename, check_type):
        print(f' => Load {check_type} checkpoint')
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(' DONE\n')


    def compute_losses(self, weight, out_weird, outputs, labels, tot_loss_ce, tot_loss_weird):
        
        loss_ce = self.loss_fn(outputs, labels)
        
        if self.LL and weight:    
            loss_weird = self.loss_weird(out_weird, loss_ce)
            loss_ce = torch.mean(loss_ce)
            loss = loss_ce + loss_weird
                    
            tot_loss_ce += loss_ce.cpu().item()
            tot_loss_weird += loss_weird.cpu().item()
        else:
            loss = torch.mean(loss_ce)
            if self.LL: tot_loss_ce += loss.item()
            
        return loss, tot_loss_ce, tot_loss_weird



    def evaluate(self, dataloader, weight):
                
        tot_loss, tot_loss_ce, tot_loss_weird, tot_accuracy = .0, .0, .0, .0
        
        self.model.eval()

        with torch.inference_mode(): # Allow inference mode
            for _, images, labels in dataloader:
                
                images, labels = images.to(self.device), labels.to(self.device)

                outputs, _, out_weird, _ = self.model(images)

                loss, tot_loss_ce, tot_loss_weird = self.compute_losses(weight, out_weird, outputs, labels, tot_loss_ce, tot_loss_weird)
                
                tot_accuracy += self.score_fn(outputs, labels)
                tot_loss += loss.item()


            tot_accuracy /= len(dataloader)
            tot_loss /= len(dataloader)
            tot_loss_ce /= len(dataloader)
            tot_loss_weird /= len(dataloader)
            
        return tot_accuracy, tot_loss, tot_loss_ce, tot_loss_weird




    def train_evaluate(self, epochs):
        
        weight = 1.
    
        check_best_path = f'{self.best_check_filename}/best_{self.method_name}.pth.tar'
		
        #best_val_loss = float('+inf')
        best_val_accuracy = 0.0
        #actual_patience = 0
        
        
        results = { 'train_loss': [], 'train_loss_ce': [], 'train_loss_weird': [], 'train_accuracy': [], 
                    'val_loss': [], 'val_loss_ce': [], 'val_loss_weird': [], 'val_accuracy': [] }
	
 
        for epoch in range(epochs):
            
            self.model.train()
            
            if epoch == 160:
                print('Decreasing learning rate to 0.01\n')
                for g in self.optimizer.param_groups: g['lr'] = 0.01
            
            if epoch == 121:
                print('Ignoring the learning loss form now\n') 
                weight = 0
            
            train_loss, train_loss_ce, train_loss_weird, train_accuracy = .0, .0, .0, .0
            
            for _, images, labels in self.lab_train_dl:

                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                                
                outputs, _, out_weird, _ = self.model(images)
                                    
                loss, train_loss_ce, train_loss_weird  = self.compute_losses(weight, out_weird, outputs, labels, train_loss_ce, train_loss_weird)
                
                loss.backward()         
                self.optimizer.step()
                                
                train_loss += loss.item()
                train_accuracy += self.score_fn(outputs, labels)

    

            train_accuracy /= self.len_lab_train_dl
            train_loss /= self.len_lab_train_dl
            train_loss_ce /= self.len_lab_train_dl
            train_loss_weird /= self.len_lab_train_dl
            
            
            val_accuracy, val_loss, val_loss_ce, val_loss_weird = self.evaluate(self.val_dl, weight)
            
            
            ###################################
            # CosineAnnealingLR
            #self.scheduler.step()
            ###################################
            

            results['train_loss'].append(train_loss)
            results['train_loss_ce'].append(train_loss_ce)
            results['train_loss_weird'].append(train_loss_weird)
            results['train_accuracy'].append(train_accuracy)
            
            
            results['val_loss'].append(val_loss)
            results['val_loss_ce'].append(val_loss_ce)
            results['val_loss_weird'].append(val_loss_weird)
            results['val_accuracy'].append(val_accuracy)
            
            
            #if(val_loss < best_val_loss):
            #    best_val_loss = val_loss
            if(val_accuracy > best_val_accuracy):
                best_val_accuracy = val_accuracy
                #actual_patience = 0
                
                print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, best_val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
                        epoch + 1, train_accuracy, train_loss, val_accuracy, best_val_accuracy, val_loss))
                
                self.__save_checkpoint(check_best_path, 'best')
                
            else:
                '''actual_patience += 1
                if actual_patience >= self.patience:
                    
                    print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, best_val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
                        epoch + 1, train_accuracy, train_loss, val_accuracy, best_val_accuracy, val_loss))
                    
                    print(f'Early stopping, validation loss do not decreased for {self.patience} epochs')
                    break'''
                
                print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, best_val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
                        epoch + 1, train_accuracy, train_loss, val_accuracy, best_val_accuracy, val_loss))
        
                    
        # load best checkpoint
        self.__load_checkpoint(check_best_path, 'best')
        

        print('Finished Training\n')
        
        return {'model_name': self.method_name, 'results': results}



    def test(self):
        test_accuracy, test_loss, test_loss_ce, test_loss_weird = self.evaluate(self.test_dl, weight=1)
        
        if test_loss_ce != 0.0:
            print('TESTING RESULTS -> test_accuracy: {:.6f}, test_loss: {:.6f}, test_loss_ce: {:.6f} , test_loss_weird: {:.6f}\n\n'.format(test_accuracy, test_loss, test_loss_ce, test_loss_weird ))
        else:
            print('TESTING RESULTS -> test_accuracy: {:.6f}, test_loss: {:.6f}\n\n'.format(test_accuracy, test_loss ))
            
        return test_accuracy, test_loss, test_loss_ce, test_loss_weird
    



    def get_new_dataloaders(self, overall_topk):
        
        # temp variable
        lab_train_indices = copy.deepcopy(self.lab_train_subset.indices)
        
        # extend with the overall_topk
        lab_train_indices.extend(overall_topk)
        
        # generate a new Subset
        self.lab_train_subset = Subset(self.transformed_trainset, lab_train_indices)
            
        # temp variable
        unlab_train_indices = copy.deepcopy(self.unlab_train_subset.indices)
        
        # remove new labeled observations
        for idx_to_remove in overall_topk: unlab_train_indices.remove(idx_to_remove)
        
        # generate a new Subset
        self.unlab_train_subset = Subset(self.non_transformed_trainset, unlab_train_indices)
        
        # sanity check
        if len(list(set(self.unlab_train_subset.indices) & set(self.lab_train_subset.indices))) == 0:
            print('Intersection between indices is EMPTY')
        else: raise Exception('NON EMPTY INDICES INTERSECTION')

        # generate the new labeled DataLoader
        self.lab_train_dl = DataLoader(
            self.lab_train_subset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True
        )
        self.len_lab_train_dl = len(self.lab_train_dl)
        

    def get_unlabebled_samples(self, unlab_sample_dim, iter):
        if(len(self.unlab_train_subset.indices) > unlab_sample_dim):
            
            # seed to impose the same sampled unlabeled subset for all strategies
            random.seed(iter)
            return random.sample(self.unlab_train_subset.indices, unlab_sample_dim)
        else:
            return self.unlab_train_subset.indices


    def train_evaluate_save(self, epochs, lab_obs, iter, results):
        
        # reinitialize the model
        self.__load_checkpoint(self.init_check_filename, 'initial')
        
        train_results = self.train_evaluate(epochs)
        
        save_train_val_curves(train_results, self.timestamp, self.dataset_name, iter, self.LL)
        
        test_accuracy, test_loss, test_loss_ce, test_loss_weird = self.test()
        
        write_csv(
            ts_dir = self.timestamp,
            dataset_name = self.dataset_name,
            head = ['method', 'lab_obs', 'test_accuracy', 'test_loss', 'test_loss_ce', 'test_loss_weird'],
            values = [self.method_name, lab_obs, test_accuracy, test_loss, test_loss_ce, test_loss_weird]
        )
        
        results['test_accuracy'].append(test_accuracy)
        results['test_loss'].append(test_loss)
        results['test_loss_ce'].append(test_loss_ce)
        results['test_loss_weird'].append(test_loss_weird)