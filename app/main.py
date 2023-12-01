# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from cifar10 import get_cifar10
from utils import get_initial_dataloaders, get_resnet18, accuracy_score, plot_loss_curves
from GTG_ActiveLearning import GTG_ActiveLearning


save_plot = True


def main():
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Assuming that we are on a CUDA machine, this should print a CUDA device

    print(f'Application running on {device}\n')

    batch_size = 64

    trainset, test_dl, classes = get_cifar10(batch_size)

    splitted_train_dl, splitted_train_ds, train_dl, val_dl = get_initial_dataloaders(trainset, 0.2, 0.1, batch_size)

    resnet18 = get_resnet18(len(classes))

    #optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)

    epochs = 30
    al_iters = 25
    top_k_obs = 1000

    GTG_AL = GTG_ActiveLearning(
        affinity_method = 'cosine_similarity', # possiible choices are: cosine_similarity, gaussian_kernel, eucliden_distance
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
    )
    


    results, n_lab_obs = GTG_AL.train_evaluate_AL_GTG(epochs=epochs, al_iters=al_iters, gtg_tol=0.001, gtg_max_iter=100, top_k_obs=top_k_obs, list_n_samples=[5, 10, 15, 20])

    plot_loss_curves(results, n_lab_obs, save_plot, f'results_{epochs}_{al_iters}_{top_k_obs}.png')
    
    
if __name__ == "__main__":
    main()