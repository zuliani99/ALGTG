
from enum import Enum
import torch

import torch.nn as nn
import torch.nn.init as init

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import numpy as np
import csv
import os
import random
from typing import List, Dict, Any, Tuple




def accuracy_score(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    outputs_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    return (outputs_class == labels).sum().item()/len(outputs)


def entropy(tensor: torch.Tensor, dim=1) -> torch.Tensor:
    x = tensor + 1e-20
    return -torch.sum(x * torch.log2(x), dim=dim)

    

def plot_loss_curves(methods_results: Dict[str, List[float]], n_lab_obs: List[int], ts_dir: str, \
    save_plot=True, plot_png_name=None) -> None:
    
    _, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (28,18))
    
    data = [
        [(0,0), 'test_loss', 'Total Loss'], [(0,1), 'test_accuracy', 'Accuracy Score'],
        [(1,0), 'test_loss_ce', 'CE Loss'], [(1,1), 'test_loss_weird', 'Loss Weird']
    ]
    
    for method_str, values in methods_results.items():
        for (pos1, pos2), key, _ in data:
            ax[pos1][pos2].plot(n_lab_obs, values[key], label = f'{method_str}')
            
    for (pos1, pos2), _, title in data:
        ax[pos1][pos2].set_title(f'{title} - # Labeled Obs')
        ax[pos1][pos2].set_xlabel('# Labeled Obs')
        ax[pos1][pos2].set_ylabel(title)
        ax[pos1][pos2].grid()
        ax[pos1][pos2].legend()
    

    plt.suptitle('Results', fontsize = 30)
    
    if save_plot: plt.savefig(f'results/{ts_dir}/{plot_png_name}')
    else: plt.show()
    
    
    
def save_train_val_curves(results_info: Dict[str, Any], ts_dir: str, dataset_name: str, al_iter: str, \
    cicle_iter: str, flag_LL: bool) -> None:

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
        
        data = [
            [(0,0), 'train_loss', 'val_loss', minloss_ep, minloss_val, 'Total Loss', 'Min Loss'],
            [(0,1), 'train_accuracy', 'val_accuracy', maxacc_ep, maxacc_val, 'Accuracy Score', 'Max Accuracy'],
            [(1,0), 'train_loss_ce', 'val_loss_ce', None, None, 'CE Loss', None],
            [(1,1), 'train_loss_weird', 'val_loss_weird',  None, None, 'Loss Weird', None]
        ]
        
        for (pos1, pos2), train_mes, val_mes, pos_mes_ep, pos_mes_val, title, label_line in data:
            ax[pos1][pos2].plot(epochs, res[train_mes], label = train_mes)
            ax[pos1][pos2].plot(epochs, res[val_mes], label = val_mes)
            if not (pos1 == 0 and pos2 ==1): ax[pos1][pos2].set_ylim([0, 5])
            
            if pos1 == 0:
                ax[pos1][pos2].axvline(pos_mes_ep, linestyle='--', color='r', label=label_line)
                ax[pos1][pos2].axhline(pos_mes_val, linestyle='--', color='r')
            
            ax[pos1][pos2].set_title(f'{title} - Epochs')
            ax[pos1][pos2].set_xlabel('Epochs')
            ax[pos1][pos2].set_ylabel(title)
            ax[pos1][pos2].grid()
            ax[pos1][pos2].legend()
        
        
    
    else:        
        data = [
            ['train_loss', 'val_loss', minloss_ep, minloss_val, 'Total Loss', 'Min Loss'],
            ['train_accuracy', 'val_accuracy', maxacc_ep, maxacc_val, 'Accuracy Score', 'Max Accuracy'],
        ]
        
        for pos, (train_mes, val_mes, pos_mes_ep, pos_mes_val, title, label_line) in enumerate(data):
            ax[pos].plot(epochs, res[train_mes], label = train_mes)
            ax[pos].plot(epochs, res[val_mes], label = val_mes)
            if pos == 0: ax[pos].set_ylim([0, 5])
            ax[pos].axvline(pos_mes_ep, linestyle='--', color='r', label=label_line)

            ax[pos].axhline(pos_mes_val, linestyle='--', color='r')
            ax[pos].set_title(f'{title} - Epochs')
            ax[pos].set_xlabel('Epochs')
            ax[pos].set_ylabel(title)
            ax[pos].grid()
            ax[pos].legend()
        

    plt.suptitle(f'AL iter {cicle_iter}.{al_iter} - {results_info["model_name"]}', fontsize = 30)
    
    path_plots = f'results/{ts_dir}/{dataset_name}/{cicle_iter}/{results_info["model_name"]}/train_val_plots/'

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


            
def create_ts_dir(timestamp: str, dataset_name: str, iter: str) -> None:
    mydir = os.path.join('results', timestamp)
    create_directory(mydir)
    create_directory(os.path.join(mydir, dataset_name))
    create_directory(os.path.join(mydir, dataset_name, iter))
    
    
    
        
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
        
    
    

def plot_gtg_entropy_tensor(tensor: torch.Tensor, topk: List[int], lab_unlabels: List[int], \
    classes: List[int], path: str, iter: int, max_x: int, dir: str) -> None:
    
    create_directory(f'{path}/gtg_entropies_plots/{dir}')
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    
    x = np.arange(max_x)
    array = tensor.cpu().numpy()
    
    new_labeled = array[topk]
    new_labeled_lab = [lab_unlabels[k] for k in topk]
    
    mask = np.ones(array.shape[0], dtype=bool)
    mask[topk] = False
    unlabeled = array[mask]

    title = 'Entropy History' if dir == 'history' else 'Entropy Derivatives'

    palette = list(mcolors.TABLEAU_COLORS.values())
    pl_cls_1, pl2_cls_2 = set(), set()

    for i, lab in zip(range(len(array)), lab_unlabels):
        if lab not in pl_cls_1:
            axes[0].plot(x, array[i], linestyle="-", label=classes[lab], color=palette[lab])
            pl_cls_1.add(lab)
        else:
            axes[0].plot(x, array[i], linestyle="-", color=palette[lab])
    
    axes[0].set_title(f'{title} Classes')
    axes[0].set_ylabel('GTG Iterations')
    axes[0].set_xlabel('Entropy')
    axes[0].grid()
    axes[0].legend()
    
    
    for array, color, style, label in [(unlabeled, 'lightblue', '--', 'unlabeled'), (new_labeled, None, '-', new_labeled_lab)]:
        if style == '-':
            labels_array = label
        
        for i in range(len(array)):
            
            if style == '-':
                color = palette[labels_array[i]]
                label = classes[labels_array[i]]
                        
            if label not in pl2_cls_2:
                axes[1].plot(x, array[i], linestyle=style, label=label, color=color)
                pl2_cls_2.add(label)
            else: axes[1].plot(x, array[i], linestyle=style, color=color)
        
    axes[1].set_ylabel('GTG Iterations')
    axes[1].set_xlabel('Entropy')
    axes[1].set_title(f'{title} - New_Lab / Unlab')
    axes[1].grid()
    axes[1].legend()
    
    plt.suptitle(f'Entropy History - Iteration {iter}', fontsize = 15)
    plt.savefig(f'{path}/gtg_entropies_plots/{dir}/{iter}.png')
    



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
    DER = 0
    H_INT = 1
    

'''class Affinity_Threshold(Enum):
    THRESHOLD = 0
    MEAN = 1
    QUANTILE = 2'''
    


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



def plot_tsne_A(A: Tuple[torch.Tensor], labels: Tuple[torch.Tensor], classes: List[str], time_stamp: str, \
    ds_name: str, samp_iter: int, method: str, affinity: str, strategy: str) -> None:
    
    A_1, A_2 = A
    
    label_lab, unlabel_lab = labels
    len_lab = len(label_lab)
    
    tsne_A1 = TSNE().fit_transform(A_1.cpu().numpy())
    tsne_A2 = TSNE().fit_transform(A_2.cpu().numpy())
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(23, 18))
        
    for idx, (tsne, name) in enumerate(zip([tsne_A1, tsne_A2], ['Uncertanity', f'{strategy} Sparsification'])):
        
        tsne_lab, tsne_unlab = tsne[:len_lab,:], tsne[len_lab:,:]
        x_lab, y_lab = tsne_lab[:,0], tsne_lab[:,1]
        x_unlab, y_unlab = tsne_unlab[:,0], tsne_unlab[:,1]
        x, y = np.hstack((x_lab, x_unlab)), np.hstack((y_lab, y_unlab))
        label = np.hstack((label_lab, unlabel_lab))
        
        sns.scatterplot(x=x_unlab, y=y_unlab, label='unlabeled', color='blue', ax=axes[idx][0])
        sns.scatterplot(x=x_lab, y=y_lab, label='labeled', color='orange', ax=axes[idx][0])
        axes[idx][0].set_title(f'{name} Affnity Matrix')
        axes[idx][0].legend()
        
        sns.scatterplot(x=x, y=y, hue=[classes[l] for l in label], ax=axes[idx][1])
        axes[idx][1].set_title(f'{name} Affnity Matrix - Classes')
        axes[idx][1].legend()

    plt.suptitle('Affinity TSNE plots')
    plt.savefig(f'results/{time_stamp}/{ds_name}/{samp_iter}/{method}/tsne_plots/{affinity}.png')
    


def plot_new_labeled_tsne(lab: Dict[str, torch.Tensor], unlab: Dict[str, torch.Tensor], iter: int, \
        method: str, ds_name: str, incides_unlab: List[int], classes: List[str], time_stamp: str, samp_iter: int):
    
    tsne = TSNE().fit_transform(np.vstack((lab['embedds'].cpu().numpy(), unlab['embedds'].cpu().numpy())))
    
    tsne_lab, tsne_unlab = tsne[:len(lab['embedds']),:], tsne[len(lab['embedds']):,:]
    label_lab, unlabel_lab = lab['labels'], unlab['labels']
    x_lab, y_lab = tsne_lab[:,0],tsne_lab[:,1]
    x_unlab, y_unlab = tsne_unlab[:,0], tsne_unlab[:,1]
    x, y = np.hstack((x_lab, x_unlab)), np.hstack((y_lab, y_unlab))
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
    plt.savefig(f'results/{time_stamp}/{ds_name}/{samp_iter}/{method}/tsne_plots/{iter - 1}.png')
    
    
    
def count_class_observation(classes, dataset, topk_idx_obs=None):
    if topk_idx_obs == None: labels = [lab for _,_,lab in dataset]
    else: labels = [dataset[k][2] for k in topk_idx_obs]
    
    d_labels = {}
    for cls in classes: d_labels[cls] = 0
    keys_d_labels = list(d_labels.keys())
    for l in labels: d_labels[keys_d_labels[l]] += 1
    return d_labels
    
    
    
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