
import torch
 
from torchvision import transforms

from termcolor import colored
from tqdm import tqdm
from utils import get_mean_std

from methods.GTG_Strategy import GTG_Strategy
from methods.Random_Strategy import Random_Strategy
from methods.Class_Entropy import Class_Entropy

from resnet_weird import LearningLoss


class ActiveLearning():

    #indices_lab_unlab_train
    def __init__(self, n_classes, batch_size, model, optimizer, train_ds, test_dl, lab_train_dl, splitted_train_ds, loss_fn, val_dl, score_fn, device, patience, timestamp): #scheduler

        self.n_classes = n_classes
        self.model = model.to(device)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.train_ds = train_ds
        self.test_dl = test_dl
        self.lab_train_dl = lab_train_dl
        self.lab_train_ds, self.unlab_train_ds = splitted_train_ds
        self.loss_fn = loss_fn
        self.val_dl = val_dl
        self.score_fn = score_fn
        self.device = device
        self.patience = patience
        self.timestamp = timestamp
        
        #resnet_weird
        self.loss_weird = LearningLoss(self.device)
        
        self.best_check_filename = 'app/checkpoints/best_checkpoint.pth.tar' #app
        self.init_check_filename = 'app/checkpoints/init_checkpoint.pth.tar' #app
        
        self.__save_checkpoint(self.init_check_filename)
        self.obtain_normalization()
        


    def reintialize_model(self):
        self.__load_checkpoint(self.init_check_filename)



    def obtain_normalization(self):
        mean, std = get_mean_std(self.lab_train_dl)
        self.normalize = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        

    def __save_checkpoint(self, filename):

        checkpoint = { 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict() }
        torch.save(checkpoint, filename)



    def __load_checkpoint(self, filename):

        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])



    def evaluate(self, val_dl, epoch = 0, epochs = 0):
        val_accuracy = .0

        self.model.eval()

        pbar = tqdm(val_dl, total = len(val_dl), leave=False)

        with torch.inference_mode(): # Allow inference mode
            for _, images, label in pbar:
                images, label = self.normalize(images.to(self.device)), label.to(self.device)

                #output = self.model(images)
                #resnet_weird
                output, _, _, _ = self.model(images)

                accuracy = self.score_fn(output, label)

                val_accuracy += accuracy

                if epoch > 0: pbar.set_description(f'EVALUATION Epoch [{epoch} / {epochs}]')
                else: pbar.set_description(f'TESTING')
                pbar.set_postfix(accuracy = accuracy)

            val_accuracy /= len(val_dl)
        return val_accuracy




    def fit(self, epochs, dataloader):
        self.model.train()
        
        
        #resnet_weird
        weight = 1.   # 120 = 0



        best_val_accuracy = -float('inf')
        actual_patience = 0

        for epoch in range(epochs):  # loop over the dataset multiple times
            
            
            #resnet_weird
            if epoch > 120:
                weight = 0
            loss_ce_total = 0
            loss_weird_total = 0
            
            

            train_accuracy = 0.0

            pbar = tqdm(enumerate(dataloader), total = len(dataloader), leave=False)

            for k, (_, images, labels) in pbar:
                                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # get the inputs; data is a list of [inputs, labels]
                images, labels = self.normalize(images.to(self.device)), labels.to(self.device)
                
                #outputs = self.model(images)
                outputs, _, out_weird, _ = self.model(images)
                
                
                #loss = self.loss_fn(outputs, labels)
                loss_ce = self.loss_fn(outputs, labels)
                
                
                #resnet_weird
                loss_weird = self.loss_weird(out_weird, loss_ce)
                loss_ce = torch.mean(loss_ce)
                
                loss = loss_ce + weight * loss_weird
                
                loss_ce_total += loss_ce
                loss_weird_total += loss_weird
                
                

                loss.backward()
                self.optimizer.step()

                accuracy = self.score_fn(outputs, labels)

                train_accuracy += accuracy

                # Update the progress bar
                pbar.set_description(f'TRAIN Epoch [{epoch + 1} / {epochs}]')
                pbar.set_postfix(accuracy = accuracy, loss_ce = loss_ce.item(), loss_weird = loss_weird.item())

            train_accuracy /= len(dataloader)

            # Validation step
            val_accuracy = self.evaluate(self.val_dl, epoch + 1, epochs)

            print('Epoch [{}], train_accuracy: {:.6f}, val_accuracy: {:.6f}, loss_ce: {:.6f}, loss_weird: {:.6f}  \n'.format(
                      epoch + 1, train_accuracy, val_accuracy, loss_ce_total.item() / k, loss_weird_total.item() / k))

            if(val_accuracy > best_val_accuracy):
                best_val_accuracy = val_accuracy
                actual_patience = 0
                self.__save_checkpoint(self.best_check_filename)
            else:
                actual_patience += 1
                if actual_patience >= self.patience:
                    print(f'Early stopping, validation loss do not decreased for {self.patience} epochs')
                    pbar.close() # Closing the progress bar before exiting from the train loop
                    break
                
                
            #resnet_weird
            if epoch == 160:
                for g in self.optimizer.param_groups:
                    g['lr'] = 0.01
                    
                    

        self.__load_checkpoint(self.best_check_filename)

        print('Finished Training\n')



    def test_AL(self):
        test_accuracy = self.evaluate(self.test_dl)

        print('\nTESTING RESULTS -> val_accuracy: {:.6f} \n'.format(test_accuracy))

        return test_accuracy



    def get_embeddings(self, type_embeds, dataloader):
            
        embeddings = torch.empty(0, self.model.linear.in_features, dtype=torch.float32).to(self.device)  
        
        self.model.eval()

        pbar = tqdm(dataloader, total = len(dataloader), leave=False, desc=f'Getting {type_embeds} Embeddings')

        # again no gradients needed
        with torch.inference_mode():
            for _, images, _ in pbar:
                
                _, embed, _, _ = self.model(self.normalize(images.to(self.device)))

                embeddings = torch.cat((embeddings, embed.squeeze()), dim=0)
             
        return embeddings



    def train_evaluate(self, epochs, al_iters, n_top_k_obs, class_entropy_params, our_method_params):

        results = { }
        n_lab_obs =  [len(self.lab_train_ds) + (iter * n_top_k_obs) for iter in range(al_iters + 1)]
        
        methods = [Class_Entropy(self, class_entropy_params), Random_Strategy(self), GTG_Strategy(self, our_method_params)]
        #methods = [GTG_Strategy(self, our_method_params)]
        
        print(colored(f'----------------------- TRAINING ACTIVE LEARNING -----------------------', 'red', 'on_white'))
        print('\n')
        
        for method in methods:
            
            print(colored(f'-------------------------- {method.method_name} --------------------------\n', 'red'))
            
            results[method.method_name] = method.run(al_iters, epochs, n_top_k_obs)
            
                    
        print(colored('Resulting dictionary', 'red', 'on_grey'))
        print(results)
        print('\n')
        
        print(colored('Resulting number of observations', 'red', 'on_grey'))
        print(n_lab_obs)
        print('\n')
        
        return results, n_lab_obs
