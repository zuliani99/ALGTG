
from enum import Enum
import torch

import torch.nn as nn
import torch.nn.init as init


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import numpy as np
import csv
import os
import random
from typing import List, Dict, Any




def accuracy_score(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    outputs_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    return (outputs_class == labels).sum().item()/len(outputs)


def entropy(tensor: torch.Tensor, dim=1) -> torch.Tensor:
    x = tensor + 1e-20
    return -torch.sum(x * torch.log2(x), dim=dim)

    

def plot_loss_curves(methods_results: Dict[str, List[float]], n_lab_obs: List[int], ts_dir: str, save_plot=True, plot_png_name=None) -> None:
    
    _, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (28,18))
    
    # test_loss
    for method_str, values in methods_results.items():
        ax[0][0].plot(n_lab_obs, values['test_loss'], label = f'{method_str}')

    ax[0][0].set_title('Total Loss - # Labeled Obs')
    ax[0][0].set_xlabel('# Labeled Obs')
    ax[0][0].set_ylabel('Total Loss')
    ax[0][0].grid()
    ax[0][0].legend()

        
    # test_accuracy
    for method_str, values in methods_results.items():
        ax[0][1].plot(n_lab_obs, values['test_accuracy'], label = f'{method_str}')
                
    ax[0][1].set_title('Accuracy Score - # Labeled Obs')
    ax[0][1].set_ylabel('Accuracy Score')
    ax[0][1].set_xlabel('# Labeled Obs')
    ax[0][1].grid()
    ax[0][1].legend()
        
        
    # test_loss_ce
    for method_str, values in methods_results.items():
        ax[1][0].plot(n_lab_obs, values['test_loss_ce'], label = f'{method_str}')
                
    ax[1][0].set_title('CE Loss - # Labeled Obs')
    ax[1][0].set_ylabel('CE Loss')
    ax[1][0].set_xlabel('# Labeled Obs')
    ax[1][0].grid()
    ax[1][0].legend()
        
        
    # test_loss_weird
    for method_str, values in methods_results.items():
        ax[1][1].plot(n_lab_obs, values['test_loss_weird'], label = f'{method_str}')
                
    ax[1][1].set_title('Loss Weird - # Labeled Obs')
    ax[1][1].set_ylabel('Loss Weird')
    ax[1][1].set_xlabel('# Labeled Obs')
    ax[1][1].grid()
    ax[1][1].legend()
    

    plt.suptitle('Results', fontsize = 30)
    
    if save_plot: plt.savefig(f'results/{ts_dir}/{plot_png_name}')
    else: plt.show()
    
    
    
def save_train_val_curves(results_info: Dict[str, Any], ts_dir: str, dataset_name: str, al_iter: str, cicle_iter: str, flag_LL: bool) -> None:

    res = results_info['results']
    epochs = range(1, len(res['train_loss']) + 1)

    if flag_LL:
        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (28,18))
    else: 
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18,8))
        
    minloss_val = min(res['val_loss'])
    minloss_ep = res['val_loss'].index(minloss_val) + 1

    maxacc_val = max(res['val_accuracy'])
    maxacc_ep = res['val_accuracy'].index(maxacc_val) + 1
    
    if flag_LL:
        
        # train_loss, val_loss
        ax[0][0].plot(epochs, res['train_loss'], label = 'train_loss')
        ax[0][0].plot(epochs, res['val_loss'], label = 'val_loss')
        ax[0][0].set_ylim([0, 5])
        ax[0][0].axvline(minloss_ep, linestyle='--', color='r', label='Min Loss')

        ax[0][0].axhline(minloss_val, linestyle='--', color='r')
        ax[0][0].set_title('Total Loss - Epochs')
        ax[0][0].set_xlabel('Epochs')
        ax[0][0].set_ylabel('Total Loss')
        ax[0][0].grid()
        ax[0][0].legend()


        # train_accuracy, val_accuracy
        ax[0][1].plot(epochs, res['train_accuracy'], label = 'train_accuracy_score')
        ax[0][1].plot(epochs, res['val_accuracy'], label = 'val_accuracy_score')
        ax[0][1].axvline(maxacc_ep, linestyle='--', color='r', label='Max Accuracy')

        ax[0][1].axhline(maxacc_val, linestyle='--', color='r')
        ax[0][1].set_title('Accuracy Score - Epochs')
        ax[0][1].set_ylabel('Accuracy Score')
        ax[0][1].set_xlabel('Epochs')
        ax[0][1].grid()
        ax[0][1].legend()
        
        
        # train_loss_ce, val_loss_ce
        ax[1][0].plot(epochs, res['train_loss_ce'], label = 'train_loss_ce')
        ax[1][0].plot(epochs, res['val_loss_ce'], label = 'val_loss_ce')
        ax[1][0].set_ylim([0, 5])

        ax[1][0].set_title('CE Loss - Epochs')
        ax[1][0].set_ylabel('CE Loss')
        ax[1][0].set_xlabel('Epochs')
        ax[1][0].grid()
        ax[1][0].legend()

        
        # train_loss_weird, val_loss_weird
        ax[1][1].plot(epochs, res['train_loss_weird'], label = 'train_loss_weird')
        ax[1][1].plot(epochs, res['val_loss_weird'], label = 'val_loss_weird')
        ax[1][1].set_ylim([0, 5])

        ax[1][1].set_title('Loss Weird - Epochs')
        ax[1][1].set_ylabel('Loss Weird')
        ax[1][1].set_xlabel('Epochs')
        ax[1][1].grid()
        ax[1][1].legend()
        
        
    
    else:
        # train_loss, val_loss
        ax[0].plot(epochs, res['train_loss'], label = 'train_loss')
        ax[0].plot(epochs, res['val_loss'], label = 'val_loss')
        ax[0].set_ylim([0, 5])
        ax[0].axvline(minloss_ep, linestyle='--', color='r', label='Min Loss')

        ax[0].axhline(minloss_val, linestyle='--', color='r')
        ax[0].set_title('Total Loss - Epochs')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Total Loss')
        ax[0].grid()
        ax[0].legend()


        # train_accuracy, val_accuracy
        ax[1].plot(epochs, res['train_accuracy'], label = 'train_accuracy_score')
        ax[1].plot(epochs, res['val_accuracy'], label = 'val_accuracy_score')
        ax[1].axvline(maxacc_ep, linestyle='--', color='r', label='Max Accuracy')

        ax[1].axhline(maxacc_val, linestyle='--', color='r')
        ax[1].set_title('Accuracy Score - Epochs')
        ax[1].set_ylabel('Accuracy Score')
        ax[1].set_xlabel('Epochs')
        ax[1].grid()
        ax[1].legend()

    plt.suptitle(f'AL iter {cicle_iter}.{al_iter} - {results_info["model_name"]}', fontsize = 30)
    
    #dataset_name
    path_plots = f'results/{ts_dir}/{dataset_name}/{cicle_iter}/train_val_plots/{results_info["model_name"]}'

    if(not os.path.exists(path_plots)):
        os.makedirs(path_plots)
    plt.savefig(os.path.join(path_plots, f'{al_iter}.png'))



def write_csv(ts_dir: str, dataset_name: str, head: List[str], values: List[str]) -> None:
    res_path = os.path.join('results', ts_dir, dataset_name, 'results.csv')
    
    if (not os.path.exists(res_path)):
        
        with open(res_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            f.close()
    
    with open(res_path, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(values)
        f.close()
                 
            

def create_directory(dir: str) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)

            
def create_ts_dir_res(timestamp: str, dataset_name: str, iter: str) -> None:
    mydir = os.path.join('results', timestamp)
    create_directory(mydir)
    create_directory(os.path.join(mydir, dataset_name))
    create_directory(os.path.join(mydir, dataset_name, iter))
    create_directory(os.path.join(mydir, dataset_name, iter, 'train_val_plots'))
    create_directory(os.path.join(mydir, dataset_name, iter, 'tsne_plots'))
    
    

        
        
# weights initiaization
def init_weights_apply(m: torch.nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=1e-3)
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
        
        
        
        
def plot_history(story_tensor: torch.Tensor, path: str, iter: int, max_x: int) -> None:
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,8))
    
    x = np.arange(max_x)

    row_sum = torch.sum(story_tensor, dim=1)
    non_zero_mask = row_sum != 0
    non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=False).squeeze()
    non_zero_row_indices = torch.unique(non_zero_indices)
    non_zero_rows = story_tensor[non_zero_row_indices]

    for i in range(len(non_zero_rows)):
        ax.plot(x, non_zero_rows[i].cpu().numpy(), linestyle="-")
        
    plt.suptitle(f'entropy {path.split("/")[3]} - iteration {iter}', fontsize = 15)
    plt.legend()
    plt.savefig(path)
    
    
    
def plot_derivatives(der_tensor: torch.Tensor, der_weighted_tensor: torch.Tensor, path: str, iter: int, max_x: int) -> None:
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20,8))
    
    x = np.arange(max_x)

    row_sum = torch.sum(der_tensor, dim=1)
    non_zero_mask = row_sum != 0
    non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=False).squeeze()
    non_zero_row_indices = torch.unique(non_zero_indices)
    non_zero_rows_der = der_tensor[non_zero_row_indices]

    for i in range(len(non_zero_rows_der)):
        ax[0].plot(x, non_zero_rows_der[i].cpu().numpy(), linestyle="-")
        
    row_sum = torch.sum(der_weighted_tensor, dim=1)
    non_zero_mask = row_sum != 0
    non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=False).squeeze()
    non_zero_row_indices = torch.unique(non_zero_indices)
    non_zero_rows_der_wei = der_weighted_tensor[non_zero_row_indices]

    for i in range(len(non_zero_rows_der_wei)):
        ax[1].plot(x, non_zero_rows_der_wei[i].cpu().numpy(), linestyle="-")
        
    plt.suptitle(f'entropy {path.split("/")[3]} - iteration {iter}', fontsize = 15)
    plt.legend()
    plt.savefig(path)
    
    


def plot_accuracy_std_mean(timestamp: str, dataset_name: str) -> None:
    df = pd.read_csv(f'results/{timestamp}/{dataset_name}/results.csv')

    df_grouped = (
        df[['method', 'lab_obs', 'test_accuracy']].groupby(['method', 'lab_obs']).agg(['mean', 'std', 'count'])
    )
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    
    df_grouped['ci'] = 1.96 * df_grouped['std'] / np.sqrt(df_grouped['count'])
    df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
    df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
    
    methods = df_grouped['method'].unique()
    shapes = ['o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v']

    # Plot each method
    plt.figure(figsize=(14, 10))
    for idx, method in enumerate(methods):
        method_data = df_grouped[df_grouped['method'] == method]
        plt.plot(method_data['lab_obs'], method_data['mean'], label=method, linestyle = 'dashed' if method in ['Random_LL', 'Random'] else 'solid')
        plt.fill_between(method_data['lab_obs'], method_data['ci_lower'], method_data['ci_upper'], alpha=0.3)
        plt.scatter(method_data['lab_obs'], method_data['mean'], marker=shapes[idx], color=plt.gca().lines[-1].get_color(), zorder=5)

    plt.xlabel('Labeled Observations')
    plt.ylabel('Test Accuracy')
    plt.title(f'{dataset_name} results')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'results/{timestamp}/{dataset_name}/mean_std_accuracy_plot.png') 
    

class Entropy_Strategy(Enum):
    MEAN_DERIVATIVES = 0
    WEIGHTED_AVERAGE_DERIVATIVES = 1
    HISTORY_INTEGRAL = 2
    LAST = 3


def set_seeds(seed: int = 10001) -> None:
    # setting seed and deterministic behaviour of pytorch for reproducibility
    # https://discuss.pytorch.org/t/determinism-in-pytorch-across-multiple-files/156269
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    


def download_tinyimagenet() -> None:
    if not os.path.exists('datasets/tiny-imagenet-200'):
        print(' => Downloading Tiny-IMAGENET Dataset')
        os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        os.system('unzip tiny-imagenet-200.zip -d datasets')
        os.remove('tiny-imagenet-200.zip')
        print(' DONE\n')
    else:
        print('Tiny-IMAGENET Dataset already downloaded')



def plot_new_labeled_tsne(lab: Dict[str, torch.Tensor], unlab: Dict[str, torch.Tensor], iter: int, \
        method: str, ds_name: str, incides_unlab: List[int], classes: List[str], time_stamp: str, samp_iter: int):
    
    tsne = TSNE().fit_transform(np.vstack((lab['embedds'].cpu().numpy(), unlab['embedds'].cpu().numpy())))
    
    tsne_lab = tsne[:len(lab['embedds']),:]
    tsne_unlab = tsne[len(lab['embedds']):,:]
    
    label_lab = lab['labels'].cpu().numpy()
    unlabel_lab = unlab['labels'].cpu().numpy()
    
    x_lab = tsne_lab[:,0]
    y_lab = tsne_lab[:,1]
    
    x_unlab = tsne_unlab[:,0]
    y_unlab = tsne_unlab[:,1]
        
    x = np.hstack((x_lab, x_unlab))
    y = np.hstack((y_lab, y_unlab))
    
    label = np.hstack((label_lab, unlabel_lab))
    
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(23, 12))
    
    sns.scatterplot(x=x_unlab, y=y_unlab, label='unlabeled', color='blue', ax=axes[0])
    sns.scatterplot(x=x_lab, y=y_lab, label='labeled', color='orange', ax=axes[0])
    sns.scatterplot(x=x_unlab[incides_unlab], y=y_unlab[incides_unlab], label='new_labeled', color='red', ax=axes[0])
    axes[0].set_title('TSNE -- labeled - unlabeled - new_labeled')
    axes[0].legend()
        
    
    sns.scatterplot(x=x, y=y, hue=[classes[l] for l in label], ax=axes[1])
    axes[1].set_title('TSNE - classes')
    axes[1].legend()
    
    plt.suptitle(f'{ds_name} - {method} - {iter - 1}')
    
    create_directory(f'results/{time_stamp}/{ds_name}/{str(samp_iter)}/tsne_plots/{method}')
    plt.savefig(f'results/{time_stamp}/{ds_name}/{samp_iter}/tsne_plots/{method}/{iter - 1}.png')
    
    
    
def download_coco_dataset() -> None:
    if not os.path.exists('datasets/coco'):
        print(' => Downloading COCO-2017 Dataset')
                
        os.system('wget -c http://images.cocodataset.org/zips/train2017.zip')
        os.system('wget -c http://images.cocodataset.org/zips/val2017.zip')
        os.system('wget -c http://images.cocodataset.org/zips/test2017.zip')

        os.system('unzip train2017.zip -d datasets/images')
        os.system('unzip val2017.zip -d datasets/images')
        os.system('unzip test2017.zip -d datasets/images')
        
        os.remove('train2017.zip')
        os.remove('test2017.zip')
        os.remove('val2017.zip')
        
        os.system('wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip')
        os.system('wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip')
        os.system('wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip')
        
        os.system('unzip annotations_trainval2017.zip')
        os.system('unzip stuff_annotations_trainval2017.zip')
        os.system('unzip image_info_test2017.zip')
        
        os.remove('annotations_trainval2017.zip')
        os.remove('stuff_annotations_trainval2017.zip')
        os.remove('image_info_test2017.zip')
    else:
        print('COCO-2017 Dataset already downloaded')