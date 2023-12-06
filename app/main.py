# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ActiveLearning import ActiveLearning
from cifar10 import get_cifar10
from utils import create_ts_dir_res, get_initial_dataloaders, get_resnet18, accuracy_score, plot_loss_curves

from datetime import datetime


save_plot = True

def main():
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Assuming that we are on a CUDA machine, this should print a CUDA device

    print(f'Application running on {device}\n')
    
    batch_size = 64

    trainset, test_dl, classes = get_cifar10(batch_size)

    splitted_train_dl, splitted_train_ds, train_dl, val_dl = get_initial_dataloaders(
        trainset = trainset,
        val_rateo = 0.2,
        labeled_ratio = 0.1,
        batch_size = batch_size
    )

    resnet18 = get_resnet18(len(classes))

    #optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)


    epochs = 2#30
    al_iters = 2#25
    n_top_k_obs = 1000
    
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    create_ts_dir_res(timestamp)


    Active_Learning_Cicle = ActiveLearning(
        n_classes = len(classes),
        batch_size=batch_size,
        model = resnet18,
        optimizer = optimizer,
        train_dl = train_dl,
        test_dl = test_dl,
        splitted_train_dl = splitted_train_dl,
        splitted_train_ds = splitted_train_ds,
        loss_fn = nn.CrossEntropyLoss(),
        val_dl = val_dl,
        score_fn = accuracy_score,
        scheduler = scheduler,
        device = device,
        patience = 10,
        timestamp = timestamp
    )
    
    
    our_method_params = {
        'gtg_tol': 0.001,
        'gtg_max_iter': 200,
        'list_n_samples': [5], #[5, 10, 15 20, 25, 30])
        'affinity_method': 'cosine_similarity',  # possiible choices are: cosine_similarity, gaussian_kernel, eucliden_distance
    }
    

    results, n_lab_obs = Active_Learning_Cicle.train_evaluate(epochs=epochs, al_iters=al_iters, n_top_k_obs=n_top_k_obs,
                                                      our_method_params=our_method_params)#, random_params=random_params)
    
    plot_loss_curves(results, n_lab_obs, save_plot, timestamp, f'results_{epochs}_{al_iters}_{n_top_k_obs}.png')
    
    
if __name__ == "__main__":
    main()