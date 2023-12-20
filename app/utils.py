import torch
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np

import csv
import os
import errno
import copy

import tqdm

from cifar10 import CIFAR10


def get_initial_dataloaders(trainset, val_rateo, labeled_ratio, batch_size):

    train_size = len(trainset) #50000

    val_size = int(train_size * val_rateo)
    train_size -= val_size

    train_data, val_data = random_split(trainset, [int(train_size), int(val_size)])

    # validation dataloader
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)

    train_data_size = len(train_data)

    # Calculate the number of samples for each split
    labeled_size = int(labeled_ratio * train_data_size)
    unlabeled_size = train_data_size - labeled_size

    # Get the dataset split
    labeled_set, unlabeled_set = random_split(train_data, [labeled_size, unlabeled_size])

    # Obtain the splitted dataloader
    labeled_train_dl = DataLoader(labeled_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return labeled_train_dl, (labeled_set, unlabeled_set), val_dl
    
    

def accuracy_score(output, label):
    output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
    return (output_class == label).sum().item()/len(output)



def entropy(tensor):
    print('ENTROPY OF TENSOR:   ', tensor)
    x = copy.deepcopy(tensor.cpu()) + 1e-20
    #x = torch.clone(tensor).cpu() + 1e-20
    return -torch.sum(x * torch.log2(x), dim=1)

    

def plot_loss_curves(methods_results, n_lab_obs, save_plot, ts_dir, plot_png_name = None):

    _, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18,8))
    
    for method_str, values in methods_results.items():
        if(isinstance(list(values.keys())[0], int)):
            for n_samples, results in values.items():
                ax[0].plot(n_lab_obs, results['test_loss'], label = f'{method_str} - {str(n_samples)} splits')
        else:
            ax[0].plot(n_lab_obs, values['test_loss'], label = f'{method_str}')

        
    ax[0].set_title('Loss - # Labeled Obs')
    ax[0].set_xlabel('# Labeled Obs')
    ax[0].set_ylabel('Loss')
    ax[0].grid()
    ax[0].legend()

    
    for method_str, values in methods_results.items():
        if(isinstance(list(values.keys())[0], int)):
            for n_samples, results in values.items():
                ax[1].plot(n_lab_obs, results['test_accuracy'], label = f'{method_str} - {str(n_samples)} splits')
        else:
            ax[1].plot(n_lab_obs, values['test_accuracy'], label = f'{method_str}')
            
    
    ax[1].set_title('Accuracy Score - # Labeled Obs')
    ax[1].set_ylabel('Accuracy Score')
    ax[1].set_xlabel('# Labeled Obs')
    ax[1].grid()
    ax[1].legend()

    plt.suptitle('Results', fontsize = 30)
    
    if save_plot: plt.savefig(f'../results/{ts_dir}/{plot_png_name}') #results/{ts_dir}/{plot_png_name}
    else: plt.show()



def write_csv(ts_dir, head, values):
    if (not os.path.exists(f'../results/{ts_dir}/results.csv')): #results/{ts_dir}/results.csv
        
        with open(f'../results/{ts_dir}/results.csv', 'w', encoding='UTF8') as f: #results/{ts_dir}/results.csv
            writer = csv.writer(f)
            writer.writerow(head)
            f.close()
    
    with open(f'../results/{ts_dir}/results.csv', 'a', encoding='UTF8') as f: #results/{ts_dir}/results.csv
        writer = csv.writer(f)
        writer.writerow(values)
        f.close()
                 
            
            
def create_ts_dir_res(timestamp):
    mydir = os.path.join('../results', timestamp) #results
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        
        

'''def get_new_dataloaders(overall_topk, lab_train_ds, unlab_train_ds, train_ds, batch_size):

    new_lab_train_ds = np.array([
        np.array([
            #idx, image if isinstance(image, np.ndarray) else image.numpy(), label
            idx, image if isinstance(image, torch.Tensor) else torch.tensor(image), label
        ], dtype=object) for idx, image, label in tqdm(lab_train_ds, total=len(lab_train_ds), leave=False, desc='Copying lab_train_ds')], dtype=object)

    new_unlab_train_ds = np.array([
        np.array([
            #idx, image if isinstance(image, np.ndarray) else image.numpy(), label
            idx, image if isinstance(image, torch.Tensor) else torch.tensor(image), label
        ], dtype=object) for idx, image, label in tqdm(unlab_train_ds, total=len(unlab_train_ds), leave=False, desc='Copying unlab_train_ds')], dtype=object)
                

    for topk_idx in tqdm(overall_topk, total=len(overall_topk), leave=False, desc='Modifing the Unlabeled Dataset'):
            
        new_lab_train_ds = np.vstack((new_lab_train_ds, np.expand_dims(
            np.array([train_ds[topk_idx if isinstance(topk_idx, int) else topk_idx.item()][0],
                    train_ds[topk_idx if isinstance(topk_idx, int) else topk_idx.item()][1],
                    train_ds[topk_idx if isinstance(topk_idx, int) else topk_idx.item()][2]
            ], dtype=object)
        , axis=0)))
            
        for idx, (i, _, _) in enumerate(new_unlab_train_ds):
            if i == topk_idx if isinstance(topk_idx, int) else topk_idx.item():
                new_unlab_train_ds[idx] = np.array([np.nan, np.nan, np.nan], dtype=object)
            # set a [np.nan np.nan] the row and the get all the row not equal to [np.nan, np.nan]
        
        
    lab_train_ds = CIFAR10(None, new_lab_train_ds)
    unlab_train_ds = CIFAR10(None, new_unlab_train_ds[
        np.array(
            [not np.isnan(row[0])
                for row in tqdm(new_unlab_train_ds, total=len(new_unlab_train_ds),
                                leave=False, desc='Obtaining the unmarked observation from the Unlabeled Dataset')
            ]
        )])
        
    lab_train_dl = DataLoader(lab_train_ds, batch_size=batch_size, shuffle=True)
    
    return lab_train_ds, unlab_train_ds, lab_train_dl'''