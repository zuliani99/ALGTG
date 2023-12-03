
import torch
import torch.nn as nn

from termcolor import colored
from tqdm import tqdm

from methods.GTG import GTG
from methods.Random import Random_Strategy


class ActiveLearning():


    def __init__(self, n_classes, batch_size, model, optimizer, train_dl, test_dl, splitted_train_dl, splitted_train_ds, loss_fn, val_dl, score_fn, scheduler, device, patience, timestamp):
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
        self.patience = patience
        self.timestamp = timestamp
        
        self.best_check_filename = './checkpoints/best_checkpoint.pth.tar' #app
        self.init_check_filename = './checkpoints/init_checkpoint.pth.tar' #app
        self.__save_checkpoint(self.init_check_filename)
        
        self.embed_model = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        
        

    def reintialize_model(self):
        self.__load_checkpoint(self.init_check_filename, 'Initial')



    def __save_checkpoint(self, filename):

        checkpoint = { 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict() }
        torch.save(checkpoint, filename)



    def __load_checkpoint(self, filename, type_load):

        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])



    def evaluate(self, val_dl, epoch = 0, epochs = 0):
        val_loss, val_accuracy = .0, .0

        self.model.eval()

        pbar = tqdm(val_dl, total = len(val_dl), leave=False)

        with torch.inference_mode(): # Allow inference mode
            for images, label in pbar:
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



    def fit(self, epochs, dataloader):
        self.model.train()

        best_val_loss = float('inf')
        actual_patience = 0

        for epoch in range(epochs):  # loop over the dataset multiple times

            train_loss = 0.0
            train_accuracy = 0.0

            pbar = tqdm(dataloader, total = len(dataloader), leave=False)

            for inputs, labels in pbar:
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

            train_loss /= len(dataloader)
            train_accuracy /= len(dataloader)

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



    def test_AL(self):
        test_loss, test_accuracy = self.evaluate(self.test_dl)

        print('\nTESTING RESULTS -> val_loss: {:.6f}, val_accuracy: {:.6f} \n'.format(test_loss, test_accuracy))

        return test_loss, test_accuracy




    def get_embeddings(self, type_embeds, dataloader):
            
        embeddings = torch.empty(0, list(self.model.children())[-1].in_features).to(self.device)

        self.embed_model.eval()

        pbar = tqdm(dataloader, total = len(dataloader), leave=False, desc=f'Getting {type_embeds} Embeddings')

        # again no gradients needed
        with torch.inference_mode():
            for inputs, _ in pbar:
                embed = self.embed_model(inputs.to(self.device))

                embeddings = torch.cat((embeddings, embed.squeeze()), dim=0)
             
        return embeddings



    def train_evaluate(self, epochs, al_iters, n_top_k_obs, our_method_params):#, random_params):

        results = { }
        n_lab_obs =  [len(self.lab_train_ds) + (iter * n_top_k_obs) for iter in range(al_iters + 1)]
        
        methods = [GTG(self, our_method_params), Random_Strategy(self)]
        
        print(colored(f'----------------------- TRAINING ACTIVE LEARNING -----------------------', 'red', 'on_white'))
        print('\n')
        
        for method in methods:
            
            print(colored(f'-------------------------- {method.method_name} --------------------------\n', 'red'))
            
            results[method.method_name] = method.run(al_iters, epochs, n_top_k_obs)
            
                    
        print(colored('Resulting dictionary', 'red', 'on_grey'))
        print('\n')
        print(results)
        print('\n')
        
        print(colored('Resulting number of observations', 'red', 'on_grey'))
        print('\n')
        print(n_lab_obs)
        print('\n')
        
        return results, n_lab_obs