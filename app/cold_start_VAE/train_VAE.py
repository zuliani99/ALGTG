import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets_creation.Classification import Cls_Dataset
from cold_start_VAE.VAE import DeepConvAutoencoder

import random
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


import logging
logger = logging.getLogger(__name__)



def make_random_split(dataset: Cls_Dataset, perc_val: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    ds_len = len(dataset)
    val_ds_len = int(ds_len * perc_val)
    
    train_ds_len = ds_len - val_ds_len
    
    train_ds, va_ds = random_split(dataset, (train_ds_len, val_ds_len))
    train_dl = DataLoader(train_ds, shuffle=True, pin_memory=True, batch_size=batch_size)
    val_dl = DataLoader(va_ds, shuffle=False, pin_memory=True, batch_size=batch_size)
    
    return train_dl, val_dl
    

def fit_ae(model: DeepConvAutoencoder, device: torch.device, train_ds: Cls_Dataset, perc_val=0.2, num_epochs=200, bs=128, lr=0.1, momentum=0.5):

    assert 0 < lr < 1 and num_epochs > 0 and bs > 0 and 0 <= momentum < 1
    
    #model = DeepConvAutoencoder(inp_side_len=dataset.image_size)
    model = model.to(device)

    # set optimizer, loss type and datasets (depending on the type of AE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-5)
    criterion = nn.MSELoss()
    
    train_dl, val_dl = make_random_split(train_ds, perc_val, bs)
    
    # training cycle
    loss = None  # just to avoid reference before assigment
    history = {'tr_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        # training
        model.train()
        tr_loss = 0

        for _, images, targets in train_dl:
            # zero the gradient
            optimizer.zero_grad()
                    
            images, targets = images.to(device), targets.to(device)
            
            _, outputs = model(images)
            # compute loss (flatten output in case of ConvAE. targets already flat)
            loss = criterion(torch.flatten(outputs, 1), images)
            tr_loss += loss.item()
            # propagate back the loss
            loss.backward()
            optimizer.step()
        
        tr_loss /= len(train_dl)
        history['tr_loss'].append(round(tr_loss, 5))

        # validation
        val_loss = evaluate(model=model, val_dl=val_dl, criterion=criterion, device=device)
        history['val_loss'].append(round(val_loss, 5))
        torch.cuda.empty_cache()

        logger.info(f'AE Epoch {epoch} | tr_loss: {round(tr_loss, 5)} | val_loss: {round(val_loss, 5)}')

        # simple early stopping mechanism
        if epoch >= 10:
            last_values = history['val_loss'][-10:]
            if (abs(last_values[-10] - last_values[-1]) <= 2e-5) or (last_values[-3] < last_values[-2] < last_values[-1]):
                return 



def evaluate(model: DeepConvAutoencoder, criterion: nn.MSELoss, device: torch.device, val_dl: DataLoader) -> float:

    model.eval()

    with torch.no_grad():
        val_loss = 0
        for _, images, targets in val_dl:
            images, targets = images.to(device), targets.to(device)
            
            _, outputs = model(images)
            
            loss = criterion(torch.flatten(outputs, 1), targets)
            val_loss += loss.item()
            
    return val_loss / len(val_dl)



def get_initial_sample(model: DeepConvAutoencoder, dl: DataLoader, device:torch.device, n_classes: int, n_obs_per_class: int) -> List[int]:
    
    model = model.to(device)
    model.eval()

    unlab_embedding = torch.empty((0, 100), dtype=torch.float32, device=torch.device('cpu'))

    with torch.no_grad():
        for _, images, targets in dl:
            images, targets = images.to(device), targets.to(device)
            embedds, _ = model(images)
            unlab_embedding = torch.cat((unlab_embedding, embedds.cpu()), dim=0)
            
    unlab_embedding = unlab_embedding.numpy()
    
    kmeans = KMeans(n_clusters=n_classes)
    kmeans.fit(unlab_embedding)
    
    # Find the closest points to each centroid
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, unlab_embedding)
    
    top_closest_points = []
    for centroid_index in closest_indices:
        distances = np.linalg.norm(unlab_embedding - kmeans.cluster_centers_[centroid_index], axis=1)
        closest_indices = np.argsort(distances)[:n_obs_per_class].tolist()
        top_closest_points.extend(unlab_embedding[closest_indices])
    
    random.shuffle(top_closest_points)
    return top_closest_points