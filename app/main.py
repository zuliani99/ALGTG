# -*- coding: utf-8 -*-

import torch

from ResNet18 import BasicBlock, ResNet_Weird, LearningLoss
from CIFAR10 import Cifar10SubsetDataloaders

#from methods.GTG_Strategy import GTG_Strategy
from methods.Random_Strategy import Random_Strategy
from methods.Class_Entropy import Class_Entropy

from utils import create_ts_dir_res, accuracy_score, plot_loss_curves, init_params_fn, save_init_checkpoint, init_params_apply

from datetime import datetime


save_plot = True
normalize_train = True


def train_evaluate(al_params, epochs, len_lab_train_ds, al_iters, n_top_k_obs, class_entropy_params, our_method_params):

    results = { }
    n_lab_obs = [len_lab_train_ds + (iter * n_top_k_obs) for iter in range(al_iters + 1)]
    
    methods = [
        Random_Strategy(al_params, LL=False),
        Random_Strategy(al_params, LL=True),  
        Class_Entropy(al_params, class_entropy_params, LL=False),
        Class_Entropy(al_params, class_entropy_params, LL=True),
        
        #GTG_Strategy(al_params, our_method_params)
    ]
    
    print(f'----------------------- TRAINING ACTIVE LEARNING -----------------------')
    print('\n')
        
    '''for method in methods:
            
        print(f'-------------------------- {method.method_name} --------------------------\n')
            
        results[method.method_name] = method.run(al_iters, epochs, n_top_k_obs)
            
                    
    print('Resulting dictionary')
    print(results)
    print('\n')
        
    print('Resulting number of observations')
    print(n_lab_obs)
    print('\n')'''
        
    return results, n_lab_obs



def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Application running on {device}\n')

    epochs = 200
    al_iters = 0#4#10 # the maximum is 36 for CIFAR10
    n_top_k_obs = 1000
    batch_size = 128
    patience = 50
        
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    create_ts_dir_res(timestamp)
    
    cifar10 = Cifar10SubsetDataloaders(batch_size, val_rateo = 0.2, labeled_ratio = 0.025, normalize_train = normalize_train)
    
    model = ResNet_Weird(BasicBlock, [2, 2, 2, 2])
    model.apply(init_params_apply)
    #init_params_fn(model)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss_werid = LearningLoss(device)
    
    save_init_checkpoint(model, optimizer, scheduler)


    al_params = {
        'cifar10': cifar10,
        'batch_size': batch_size,
        'score_fn': accuracy_score,
        'device': device,
        'patience': patience,
        'timestamp': timestamp,
        'loss_fn': loss_fn,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_weird': loss_werid
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
    
    if normalize_train:
        final_plot_name = f'results_{epochs}_{al_iters}_{n_top_k_obs}_NormTrain.png'
    else: 
        final_plot_name = f'results_{epochs}_{al_iters}_{n_top_k_obs}_NormLab.png'
    
    plot_loss_curves(results, n_lab_obs, save_plot, timestamp, normalize_train, final_plot_name)
    

if __name__ == "__main__":
    main()