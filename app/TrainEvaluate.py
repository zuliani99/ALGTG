
import torch
from torch.utils.data import DataLoader, Subset

import copy

from CIFAR10 import CIFAR10, Cifar10SubsetDataloaders
from utils import save_train_val_curves, write_csv


class TrainEvaluate(object):

    def __init__(self, params, LL):
        
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
        
        
        #I need the deep copy only of the subsets, that have the indices referred to the original_trainset
        self.lab_train_subset: Subset = copy.deepcopy(cifar10.lab_train_subset)
        self.unlab_train_subset: Subset = copy.deepcopy(cifar10.unlab_train_subset)
        
        #parameters that are used for all the strategies
        self.lab_train_dl = DataLoader(
            self.lab_train_subset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True
        )
        self.len_lab_train_dl = len(self.lab_train_dl)
        self.test_dl: DataLoader = cifar10.test_dl
        self.val_dl: DataLoader = cifar10.val_dl
        
        self.transformed_trainset: CIFAR10 = cifar10.transformed_trainset 
        self.non_transformed_trainset: CIFAR10 = cifar10.non_transformed_trainset 
        
        self.model = params['model'].to(self.device)
        self.optimizer = params['optimizer']
        self.scheduler = params['scheduler']
        self.loss_weird = params['loss_weird']
        
        self.LL = LL
        

        
    def __save_best_checkpoint(self, filename):
        print(' => Saving best checkpoint')
        checkpoint = {
            'state_dict': self.model.state_dict(), 
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, filename)
        print(' DONE\n')
    
    
    
    def reintialize_model(self):
        print(' => Load initial checkpoint')
        checkpoint = torch.load(self.init_check_filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(' DONE\n')
        


    def __load_best_checkpoint(self, filename):
        print(' => Loading best checkpoint')
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
            
        return loss



    def evaluate(self, dataloader, weight):
                
        tot_loss, tot_loss_ce, tot_loss_weird, tot_accuracy = .0, .0, .0, .0
        
        self.model.eval()

        with torch.inference_mode(): # Allow inference mode
            for _, images, labels in dataloader:
                
                images, labels = images.to(self.device), labels.to(self.device)

                outputs, _, out_weird, _ = self.model(images)

                loss = self.compute_losses(weight, out_weird, outputs, labels, tot_loss_ce, tot_loss_weird)
                
                tot_accuracy += self.score_fn(outputs, labels)
                tot_loss += loss.item()


            tot_accuracy /= len(dataloader)
            tot_loss /= len(dataloader)
            tot_loss_ce /= len(dataloader)
            tot_loss_weird /= len(dataloader)
            
        return tot_accuracy, tot_loss, tot_loss_ce, tot_loss_weird




    def train_evaluate(self, epochs, method_str):
        
        weight = 1.   # 120 = 0
    
        check_best_path = f'{self.best_check_filename}/best_{method_str}.pth.tar'
		
        best_val_accuracy = float('-inf')
        actual_patience = 0
        
        
        results = { 'train_loss': [], 'train_loss_ce': [], 'train_loss_weird': [], 'train_accuracy': [], 
                    'val_loss': [], 'val_loss_ce': [], 'val_loss_weird': [], 'val_accuracy': [] }
	
 
        for epoch in range(epochs):
            
            self.model.train()
            
            if epoch == 160:
                print('Decreasing learning rate to 0.01 and ignoring the learning loss\n')
                for g in self.optimizer.param_groups: g['lr'] = 0.01
            
            # > 120 set weight = 0
            if epoch == 121: weight = 0
            
            train_loss, train_loss_ce, train_loss_weird, train_accuracy = .0, .0, .0, .0
            
            for _, images, labels in self.lab_train_dl:

                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                                
                outputs, _, out_weird, _ = self.model(images)
                                    
                loss = self.compute_losses(weight, out_weird, outputs, labels, train_loss_ce, train_loss_weird)
                
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
            
            
            
            if(val_accuracy > best_val_accuracy):
                best_val_accuracy = val_accuracy
                actual_patience = 0
                
                print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, best_val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
                        epoch + 1, train_accuracy, train_loss, val_accuracy, best_val_accuracy, val_loss))
                
                self.__save_best_checkpoint(check_best_path)
                
            else:
                actual_patience += 1
                if actual_patience >= self.patience:
                    
                    print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, best_val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
                        epoch + 1, train_accuracy, train_loss, val_accuracy, best_val_accuracy, val_loss))
                    
                    print(f'Early stopping, validation loss do not decreased for {self.patience} epochs')
                    break
                
                print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, best_val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
                        epoch + 1, train_accuracy, train_loss, val_accuracy, best_val_accuracy, val_loss))    
                    

        self.__load_best_checkpoint(check_best_path)

        print('Finished Training\n')
        
        return {'model_name': method_str, 'results': results}



    def test(self):
        test_accuracy, test_loss, test_loss_ce, test_loss_weird = self.evaluate(self.test_dl, weight=1)
        
        if test_loss_ce != 0.0:
            print('TESTING RESULTS -> test_accuracy: {:.6f}, test_loss: {:.6f}, test_loss_ce: {:.6f} , test_loss_weird: {:.6f}\n\n'.format(test_accuracy, test_loss, test_loss_ce, test_loss_weird ))
        else:
            print('TESTING RESULTS -> test_accuracy: {:.6f}, test_loss: {:.6f}\n\n'.format(test_accuracy, test_loss ))
            
        return test_accuracy, test_loss, test_loss_ce, test_loss_weird



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
        


    def train_evaluate_save(self, epochs, lab_obs, n_splits, iter, results):
        
        # reinitialize the model
        self.reintialize_model()
        
        train_results = self.train_evaluate(epochs, self.method_name)
        
        save_train_val_curves(train_results, self.timestamp, iter, self.LL)
        
        test_accuracy, test_loss, test_loss_ce, test_loss_weird = self.test()
        
        write_csv(
            ts_dir = self.timestamp,
            head = ['method', 'lab_obs', 'n_splits', 'test_accuracy', 'test_loss', 'test_loss_ce', 'test_loss_weird'],
            values = [self.method_name, lab_obs, str(n_splits), test_accuracy, test_loss, test_loss_ce, test_loss_weird]
        )
        
        if n_splits == None:
            results['test_accuracy'].append(test_accuracy)
            results['test_loss'].append(test_loss)
            results['test_loss_ce'].append(test_loss_ce)
            results['test_loss_weird'].append(test_loss_weird)
        else:
            results[n_splits]['test_accuracy'].append(test_accuracy)
            results[n_splits]['test_loss'].append(test_loss)
            results[n_splits]['test_loss_ce'].append(test_loss_ce)
            results[n_splits]['test_loss_weird'].append(test_loss_weird)
            