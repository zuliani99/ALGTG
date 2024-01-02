# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from termcolor import colored

#from ActiveLearning import ActiveLearning

from resnet.resnet_weird import ResNet_Weird, BasicBlock
from resnet.resnet18 import ResNet18

from methods.GTG_Strategy import GTG_Strategy
from methods.Random_Strategy import Random_Strategy
from methods.Class_Entropy import Class_Entropy

from cifar10 import get_cifar10
from utils import create_ts_dir_res, get_initial_dataloaders, accuracy_score, plot_loss_curves, init_params

from datetime import datetime


save_plot = True
use_resnet_weird = True



def train_evaluate(al_params, epochs, len_lab_train_ds, al_iters, n_top_k_obs, class_entropy_params, our_method_params):

    results = { }
    n_lab_obs =  [len_lab_train_ds + (iter * n_top_k_obs) for iter in range(al_iters + 1)]
       
    methods = [Class_Entropy(al_params, class_entropy_params), Random_Strategy(al_params), GTG_Strategy(al_params, our_method_params)]
        
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
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Application running on {device}\n')
    
    batch_size = 128
    patience = 20

    original_trainset, test_dl, classes = get_cifar10(batch_size)
    
    #indices_lab_unlab_train
    lab_train_dl, splitted_train_ds, val_dl = get_initial_dataloaders(
        trainset = original_trainset,
        val_rateo = 0.2,
        labeled_ratio = 0.025, # like vascon experiment
        batch_size = batch_size
    )

    if use_resnet_weird:
        #resnet_weird
        resnet18 = ResNet_Weird(BasicBlock, [2, 2, 2, 2], num_classes=len(classes))
        
        # weights initiaization
        resnet18.apply(init_params)
        
        cross_entropy = nn.CrossEntropyLoss(reduction='none')
    else:
        #normal resnet
        resnet18 = ResNet18(len(classes))
        
        # weights initiaization
        resnet18.apply(init_params)
        
        cross_entropy = nn.CrossEntropyLoss()
        
        

    optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)
    
    #optimizer = torch.optim.AdamW(resnet18.parameters(), lr=0.001)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, verbose=True)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

    epochs = 100
    al_iters = 5 # the maximum is 36
    n_top_k_obs = 1000
    
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    create_ts_dir_res(timestamp)


    al_params = {
        'n_classes': len(classes),
        'batch_size': batch_size,
        'model': resnet18,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_ds': original_trainset,
        'test_dl': test_dl,
        'lab_train_dl': lab_train_dl,
        'splitted_train_ds': splitted_train_ds,
        'loss_fn': cross_entropy,
        'val_dl': val_dl,
        'score_fn': accuracy_score,
        'device': device,
        'patience': patience,
        'timestamp': timestamp, 
    }    
    
    
    our_method_params = {
        'gtg_tol': 0.001,
        'gtg_max_iter': 20,
        'list_n_samples': [10], #[5, 10, 15 20, 25, 30])
    }
    
    class_entropy_params = {
        'list_n_samples': [10], #[5, 10, 15 20, 25, 30])
    }
    
                                                      
    results, n_lab_obs = train_evaluate(al_params=al_params, epochs=epochs, len_lab_train_ds=len(splitted_train_ds[0]), al_iters=al_iters, n_top_k_obs=n_top_k_obs,
                                        class_entropy_params=class_entropy_params, our_method_params=our_method_params)
    
    plot_loss_curves(results, n_lab_obs, save_plot, timestamp, f'results_{epochs}_{al_iters}_{n_top_k_obs}.png')
    
    
if __name__ == "__main__":
    main()
