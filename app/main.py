# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ActiveLearning import ActiveLearning

from resnet.resnet_weird import ResNet_Weird, BasicBlock
from resnet.resnet18 import ResNet18

from cifar10 import get_cifar10
from utils import create_ts_dir_res, get_initial_dataloaders, accuracy_score, plot_loss_curves

from datetime import datetime


save_plot = True
use_resnet_weird = True

def main():
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print(f'Application running on {device}\n')
    
    batch_size = 128

    original_trainset, test_dl, classes = get_cifar10(batch_size)
    
    #indices_lab_unlab_train
    lab_train_dl, splitted_train_ds, val_dl = get_initial_dataloaders(
        trainset = original_trainset,
        val_rateo = 0.2,
        labeled_ratio = 0.01,
        batch_size = batch_size
    )

    if use_resnet_weird:
        #resnet_weird
        resnet18 = ResNet_Weird(BasicBlock, [2, 2, 2, 2], num_classes=len(classes))
        cross_entropy = nn.CrossEntropyLoss(reduction='none')
    else:
        #normal resnet
        resnet18 = ResNet18(len(classes))
        cross_entropy = nn.CrossEntropyLoss()
    
    
    #optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    #optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)


    epochs = 50
    al_iters = 36#25
    n_top_k_obs = 1000
    
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    create_ts_dir_res(timestamp)


    Active_Learning_Cicle = ActiveLearning(
        n_classes = len(classes),
        batch_size = batch_size,
        model = resnet18,
        optimizer = optimizer,
        train_ds = original_trainset, #50000
        test_dl = test_dl,
        lab_train_dl = lab_train_dl,
        splitted_train_ds = splitted_train_ds,
        loss_fn = cross_entropy,
        val_dl = val_dl,
        score_fn = accuracy_score,
        device = device,
        patience = 10,
        timestamp = timestamp
    )
    
    
    our_method_params = {
        'gtg_tol': 0.001,
        'gtg_max_iter': 20,#200,
        'list_n_samples': [10], #[5, 10, 15 20, 25, 30])
    }
    
    class_entropy_params = {
        'list_n_samples': [10], #[5, 10, 15 20, 25, 30])
    }
    

    results, n_lab_obs = Active_Learning_Cicle.train_evaluate(epochs=epochs, al_iters=al_iters, n_top_k_obs=n_top_k_obs,
                                                      class_entropy_params=class_entropy_params,
                                                      our_method_params=our_method_params)
    
    plot_loss_curves(results, n_lab_obs, save_plot, timestamp, f'results_{epochs}_{al_iters}_{n_top_k_obs}.png')
    
    
if __name__ == "__main__":
    main()
