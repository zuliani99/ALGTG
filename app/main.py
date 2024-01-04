# -*- coding: utf-8 -*-

import torch
#import torch.nn as nn

from termcolor import colored

from ResNet18 import BasicBlock, ResNet_Weird
from CIFAR10 import Cifar10SubsetDataloaders

#from ActiveLearning import ActiveLearning

#from resnet.resnet_weird import ResNet_Weird, BasicBlock
#from resnet.resnet18 import ResNet18

from methods.GTG_Strategy import GTG_Strategy
from methods.Random_Strategy import Random_Strategy
from methods.Class_Entropy import Class_Entropy

#from Active_Learning_GTG.app.CIFAR10 import get_cifar10
from utils import create_ts_dir_res, accuracy_score, plot_loss_curves#, init_params, init_params2, get_initial_dataloaders,

from datetime import datetime


save_plot = True
#use_resnet_weird = True



def train_evaluate(al_params, epochs, len_lab_train_ds, al_iters, n_top_k_obs, class_entropy_params, our_method_params):

    results = { }
    n_lab_obs = [len_lab_train_ds + (iter * n_top_k_obs) for iter in range(al_iters + 1)]
       
    methods = [Class_Entropy(al_params, class_entropy_params), Random_Strategy(al_params), GTG_Strategy(al_params, our_method_params)]
    #methods = [Random_Strategy(al_params)]
    
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



def main():
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print(f'Application running on {device}\n')
    
    batch_size = 128
    patience = 30

    '''original_trainset, test_dl, classes = get_cifar10(batch_size)
    
    #indices_lab_unlab_train
    lab_train_dl, splitted_train_ds, val_dl = get_initial_dataloaders(
        trainset = original_trainset,
        val_rateo = 0.2,
        labeled_ratio = 0.025, # like vascon experiment
        batch_size = batch_size
    )
    '''
    #if use_resnet_weird:
        #resnet_weird
    #    resnet18 = ResNet_Weird(BasicBlock, [2, 2, 2, 2])#, num_classes=len(classes))
    #else:
        #normal resnet
    #    resnet18 = ResNet18(len(classes))
        
    #resnet18.apply(init_params)
    #init_params2(resnet18)
    #cross_entropy = nn.CrossEntropyLoss()
    

    #optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, verbose=True)


    epochs = 100
    al_iters = 5#10 # the maximum is 36
    n_top_k_obs = 1000
    
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    create_ts_dir_res(timestamp)
    
    cifar10 = Cifar10SubsetDataloaders(batch_size)
    cifar10.get_initial_dataloaders(val_rateo = 0.2, labeled_ratio = 0.025)
    
    model = ResNet_Weird(BasicBlock, [2, 2, 2, 2])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, verbose=True)
    #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True, patience=7)
    #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.65 ** epoch, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-3, last_epoch=-1, verbose=True)


    al_params = {
        
        
        # deep copy
        #'model': resnet18,
        #'optimizer': optimizer,
        #'scheduler': scheduler, # da cancellare qui
        #'lab_train_dl': lab_train_dl,
        #'splitted_train_ds': splitted_train_ds,
        
        #'train_ds': original_trainset,
        #'test_dl': test_dl,
        #'val_dl': val_dl,
        # shallow
        #'n_classes': len(classes),
        #'loss_fn': cross_entropy,
        'cifar10': cifar10,
        'batch_size': batch_size,
        'score_fn': accuracy_score,
        'device': device,
        'patience': patience,
        'timestamp': timestamp,
        'loss_fn': torch.nn.CrossEntropyLoss(),
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler
    }    
    
    
    our_method_params = {
        'gtg_tol': 0.001,
        'gtg_max_iter': 20,
        'list_n_samples': [10], #[5, 10, 15 20, 25, 30])
    }
    
    class_entropy_params = {
        'list_n_samples': [10], #[5, 10, 15 20, 25, 30])
    }
    
                                                      
    results, n_lab_obs = train_evaluate(al_params=al_params, epochs=epochs, len_lab_train_ds=len(cifar10.lab_train_subset),
                                        al_iters=al_iters, n_top_k_obs=n_top_k_obs,
                                        class_entropy_params=class_entropy_params, our_method_params=our_method_params)
    
    plot_loss_curves(results, n_lab_obs, save_plot, timestamp, f'results_{epochs}_{al_iters}_{n_top_k_obs}.png')
    
    
if __name__ == "__main__":
    main()
