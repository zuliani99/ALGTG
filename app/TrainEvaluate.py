
import torch
from torch.utils.data import DataLoader, Subset

from torchvision.transforms import v2
from torchvision.utils import save_image

from tqdm import tqdm
import copy
from CIFAR10 import CIFAR10, Cifar10SubsetDataloaders
from utils import get_mean_std, init_params

import os


#from resnet.resnet_weird import LearningLoss

class TrainEvaluate(object):

    def __init__(self, params):
        
        self.best_check_filename = 'app/checkpoints'
        self.init_check_filename = 'app/checkpoints/init_checkpoint.pth.tar'
        
        self.cifar10: Cifar10SubsetDataloaders = params['cifar10']
        
        self.n_classes = len(self.cifar10.classes)
        self.device = params['device']
        self.batch_size = params['batch_size']
        self.score_fn = params['score_fn']
        self.patience = params['patience']
        self.timestamp = params['timestamp']
        self.loss_fn = params['loss_fn']
        
        self.lab_train_dl: DataLoader = copy.deepcopy(self.cifar10.lab_train_dl)
        self.lab_train_subset: Subset = copy.deepcopy(self.cifar10.lab_train_subset)
        self.unlab_train_subset: Subset = copy.deepcopy(self.cifar10.unlab_train_subset)
        
        self.test_dl: DataLoader = self.cifar10.test_dl
        self.val_dl: DataLoader = self.cifar10.val_dl
        self.original_trainset: CIFAR10 = self.cifar10.original_trainset

        self.original_trainset.lab_train_idxs = self.lab_train_subset.indices
        
        self.model = params['model'].to(self.device)
        self.optimizer = params['optimizer']
        self.scheduler = params['scheduler']
        self.model.apply(init_params)
        
        
        if not os.path.exists(self.init_check_filename):
            self.__save_init_checkpoint()
        else:
            self.__load_init_checkpoint()
                  
        #self.__save_init_checkpoint()
        self.obtain_normalization()
        
        #resnet_weird
        #self.loss_weird = LearningLoss(self.device)
        
        


    def reintialize_model(self):
        self.__load_init_checkpoint()



    def obtain_normalization(self):
        mean, std = get_mean_std(self.lab_train_dl)
        self.normalize = v2.Compose([ v2.Normalize(mean, std) ])
        


    def __save_init_checkpoint(self):

        checkpoint = { 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict() }
        torch.save(checkpoint, self.init_check_filename)
        
        
        
    def __save_best_checkpoint(self, filename, actual_patience, epoch, best_val_loss):

        checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                      'actual_patience': actual_patience, 'epoch': epoch, 'best_val_loss': best_val_loss}
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
        
        return checkpoint['actual_patience'], checkpoint['epoch'], checkpoint['best_val_loss']



    def evaluate(self, val_dl, epoch = 0, epochs = 0):
        val_accuracy, val_loss = .0, .0

        self.model.eval()

        pbar = tqdm(val_dl, total = len(val_dl), leave=False)

        with torch.inference_mode(): # Allow inference mode
            for idxs, images, labels in pbar:
                
                images, labels = self.normalize(images.to(self.device)), labels.to(self.device)
                #images, labels = images.to(self.device), labels.to(self.device)
                
                #save_image(images[0], f'images/img{idxs[0]}.png')

                #if self.model.__class__.__name__ == 'ResNet_Weird':
                    #resnet_weird
                outputs, _, _, _ = self.model(images)
                #else:
                    #output = self.model(images)
                
                #loss = torch.mean(self.loss_fn(output, label)).item()
                loss = self.loss_fn(outputs, labels).item()
                accuracy = self.score_fn(outputs, labels)

                val_accuracy += accuracy
                val_loss += loss

                if epoch > 0: pbar.set_description(f'EVALUATION Epoch [{epoch} / {epochs}]')
                else: pbar.set_description(f'TESTING')
                pbar.set_postfix(accuracy = accuracy, loss = loss)

            val_accuracy /= len(val_dl)
            val_loss /= len(val_dl)
            
        return val_accuracy, val_loss




    def fit(self, epochs, dataloader, method_str):
        self.model.train()
        
        
        #resnet_weird
        #weight = 1.   # 120 = 0



        check_best_path = f'{self.best_check_filename}/best_{method_str}.pth.tar'
		
        #actual_epoch = 0
        best_val_loss = float('inf')
        actual_patience = 0

        #if os.path.exists(check_best_path):
        #    actual_patience, actual_epoch, best_val_loss = self.__load_best_checkpoint(check_best_path)
		
	
        self.model.train()

        for epoch in range(epochs):#(actual_epoch, epochs):  # loop over the dataset multiple times
            
            
            #resnet_weird
            #if epoch > 120:
            #    weight = 0
            #loss_weird_total = 0
            
            
            train_loss = 0.0
            train_accuracy = 0.0
            
            #iters = len(dataloader)

            #pbar = tqdm(enumerate(dataloader), total = iters, leave=False)
            pbar = tqdm(dataloader, total = len(dataloader), leave=False)

            #for batch_idx, (_, images, labels) in pbar:
            for idxs, images, labels in pbar:
                                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # get the inputs; data is a list of [inputs, labels]
                images, labels = self.normalize(images.to(self.device)), labels.to(self.device)
                #images, labels = images.to(self.device), labels.to(self.device)
                
                #save_image(images[0], f'images/img{idxs[0]}.png')
                
                #if self.model.__class__.__name__ == 'ResNet_Weird':
                    #resnet_weird               
                
                    #outputs = self.model(images)
                    #outputs, _, out_weird, _ = self.model(images)
                    #outputs, _, out_weird, _ = self.model(images)
                    
                    
                    #loss = self.loss_fn(outputs, labels)
                    #loss_ce = self.loss_fn(outputs, labels)
                    
                    
                    #resnet_weird
                    #loss_weird = self.loss_weird(out_weird, loss_ce)
                    #loss_ce = torch.mean(loss_ce)
                    
                    #loss = loss_ce + weight * loss_weird
                    
                    #train_loss += loss_ce
                    #loss_weird_total += loss_weird
                
                #else:
                    #outputs = self.model(images)
                    
                outputs, _, _, _ = self.model(images)
                
                
                #loss = torch.mean(self.loss_fn(outputs, labels))
                loss = self.loss_fn(outputs, labels)
                train_loss += loss.item()
                

                loss.backward()
                
                #nn.utils.clip_grad_norm_(self.model.parameters(), 1., norm_type=2)
                
                self.optimizer.step()
                
                
                # cosine annealing with warm restart
                #self.scheduler.step(epoch + batch_idx / iters)
                
                

                accuracy = self.score_fn(outputs, labels)

                train_accuracy += accuracy

                # Update the progress bar
                pbar.set_description(f'TRAIN Epoch [{epoch + 1} / {epochs}]')
                #if self.model.__class__.__name__ == 'ResNet_Weird':
                #    pbar.set_postfix(accuracy = accuracy, loss_ce = loss_ce.item(), loss_weird = loss_weird.item())
                #else:
                pbar.set_postfix(accuracy = accuracy, loss = loss.item())
    

            train_accuracy /= len(dataloader)
            train_loss /= len(dataloader)
            
			# scheduler step
            #self.scheduler.step(train_loss)
            
            # CosineAnnealingLR
            self.scheduler.step()

            # Validation step
            val_accuracy, val_loss = self.evaluate(self.val_dl, epoch + 1, epochs)
            
            
            
            # ReduceLROnPlateau
            #self.scheduler.step(val_loss)
            #self.scheduler.step()
            
            
            
            

            #if self.model.__class__.__name__ == 'ResNet_Weird':
            #    print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, loss_weird: {:.6f}, val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
            #          epoch + 1, train_accuracy, train_loss, loss_weird_total / len(dataloader), val_accuracy, val_loss))
            #else:
            print('Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, val_loss: {:.6f}, best_val_loss: {:.6f}\n'.format(
                      epoch + 1, train_accuracy, train_loss, val_accuracy, val_loss, val_loss if val_loss < best_val_loss else best_val_loss))


            if(val_loss < best_val_loss):
                best_val_loss = val_loss
                actual_patience = 0
                self.__save_best_checkpoint(check_best_path, actual_patience, epoch, best_val_loss)
            else:
                actual_patience += 1
                if actual_patience >= self.patience:
                    print(f'Early stopping, validation loss do not decreased for {self.patience} epochs')
                    pbar.close() # Closing the progress bar before exiting from the train loop
                    break
                
                
            #resnet_weird
            #if epoch == 160:
            #    for g in self.optimizer.param_groups:
            #        g['lr'] = 0.01
                    
                    

        self.__load_best_checkpoint(check_best_path)

        print('Finished Training\n')



    def test_AL(self):
        test_accuracy, test_loss = self.evaluate(self.test_dl)

        print('\nTESTING RESULTS -> test_accuracy: {:.6f}, test_loss: {:.6f} \n'.format(test_accuracy, test_loss))

        return test_accuracy, test_loss



    def get_embeddings(self, type_embeds, dataloader):
        #if self.model.__class__.__name__ == 'ResNet_Weird':
        embeddings = torch.empty(0, self.model.linear.in_features, dtype=torch.float32).to(self.device)  
        self.model.eval()
        #else:
        #    embeddings = torch.empty(0, list(self.model.resnet18.children())[-1].in_features, dtype=torch.float32).to(self.device)  
        #    embed_model = nn.Sequential(*list(self.model.resnet18.children())[:-1]).to(self.device)
        #    embed_model.eval()
            
        pbar = tqdm(dataloader, total = len(dataloader), leave=False, desc=f'Getting {type_embeds} Embeddings')

        # again no gradients needed
        with torch.inference_mode():
            for _, images, _ in pbar:
                
                images = self.normalize(images.to(self.device))
                #images = images.to(self.device)
                
                #if self.model.__class__.__name__ == 'ResNet_Weird':
                _, embed, _, _ = self.model(images)
                #else:
                #    embed = embed_model(images)
                

                embeddings = torch.cat((embeddings, embed.squeeze()), dim=0)
             
        return embeddings



    def get_new_dataloaders(self, overall_topk):
        
        lab_train_indices = self.lab_train_subset.indices
        
        lab_train_indices.extend(overall_topk)
        self.lab_train_subset = Subset(self.original_trainset, lab_train_indices)
        
        
        # update the indices for the transform        
        self.original_trainset.lab_train_idxs = lab_train_indices
        

        unlab_train_indices = self.unlab_train_subset.indices
        for idx_to_remove in overall_topk:
            unlab_train_indices.remove(idx_to_remove)
        self.unlab_train_subset = Subset(self.original_trainset, unlab_train_indices)
        
        if len(list(set(self.unlab_train_subset.indices) & set(self.lab_train_subset.indices))) == 0:
            print('Intersection between indices is EMPTY')
        else: raise Exception('NON EMPTY INDICES INTERSECTION')

        self.lab_train_dl = DataLoader(self.lab_train_subset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.obtain_normalization()
