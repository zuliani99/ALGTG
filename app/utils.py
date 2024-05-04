
from matplotlib.typing import ColorType
import torch
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.init as init

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import numpy as np
import csv
import os
import random
from typing import List, Dict, Any, Tuple



import logging
logger = logging.getLogger(__name__)



def accuracy_score(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    outputs_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    return (outputs_class == labels).sum().item()/len(outputs)


def entropy(tensor: torch.Tensor, dim=1) -> torch.Tensor:
    x = tensor + 1e-20
    return -torch.sum(x * torch.log2(x), dim=dim)


def plot_trail_acc(dataset_name: str, trial: int, methods_results: Dict[str, Dict[str, List[float]]], n_lab_obs: List[int], ts_dir: str, \
    key: str, save_plot=True, plot_png_name=None) -> None:
    
    if key == 'test_map': measure = 'mAP'
    else: measure = 'Accuracy'
    
    plt.figure(figsize = (14,10))
    
    shapes = ['o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v']
    palette = get_palette(len(methods_results.items()))
    lines_handles = []
    
    for idx, (method_str, values) in enumerate(methods_results.items()):
        plt.plot(n_lab_obs, values[key], label = f'{method_str}',
                                     linestyle = 'dashed' if method_str.split('_')[1] in 'Random' else 'solid', color=palette[idx])
        plt.scatter(n_lab_obs, values[key], marker=shapes[idx], color=palette[idx], zorder=5)
        lines_handles.append(mlines.Line2D([], [], color=palette[idx], marker=shapes[idx], markersize=5, label=method_str))
  
    plt.title(f'{dataset_name} - Trial {trial} - {measure} Labeled Obs', fontsize = 30)
    plt.xlabel('# Labeled Obs', fontsize = 15)
    plt.ylabel(measure, fontsize = 15)
    plt.grid(True)
    plt.legend(handles=lines_handles)
    
    if save_plot: plt.savefig(f'results/{ts_dir}/{plot_png_name}')
    else: plt.show()
    
    
    
def save_train_val_curves(list_dict_keys: List[str], results_info: Dict[str, Any], strategy_name: str, \
                          ts_dir: str, dataset_name: str, al_iter: int, cicle_iter: int) -> None:

    epochs = range(1, len(results_info[list_dict_keys[0]]) + 1)

    _, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (28,18))
        
    data = zip([(0,0), (0,1), (1,0), (1,1)], list_dict_keys)
        
    for (pos1, pos2), train_mes in data:
        ax[pos1][pos2].plot(epochs, results_info[train_mes], label = train_mes)            
        ax[pos1][pos2].set_title(f'{train_mes} - Epochs', fontsize = 15)
        ax[pos1][pos2].set_xlabel('Epochs', fontsize = 10)
        ax[pos1][pos2].set_ylabel(train_mes, fontsize = 10)
        ax[pos1][pos2].grid()
        ax[pos1][pos2].legend()
        

    plt.suptitle(f'Iteration: {al_iter} - {strategy_name}', fontsize = 30)
    plt.savefig(f'results/{ts_dir}/{dataset_name}/{cicle_iter}/{strategy_name}/train_val_plots/{al_iter}.png')



def print_cumulative_train_results(list_dict_keys: List[str], cum_train_results: Dict[str, Any], strategy_name: str,\
                                   epochs: int, ts_dir: str, dataset_name: str, cicle_iter: int):

    _, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (28,18))
        
    data = zip([(0,0), (0,1), (1,0), (1,1)], list_dict_keys)
    x = range(1, epochs + 1)
    
    palette = get_palette(len(cum_train_results.items()))
    
    for (pos1, pos2), train_mes in data:
        for idx, (iter, results_info) in enumerate(cum_train_results.items()):
            ax[pos1][pos2].plot(x, results_info[train_mes], label = f'{train_mes}_{iter}', color=palette[idx])
                        
        ax[pos1][pos2].set_title(f'{train_mes} - Epochs', fontsize = 15)
        ax[pos1][pos2].set_xlabel('Epochs', fontsize = 10)
        ax[pos1][pos2].set_ylabel(train_mes, fontsize = 10)
        ax[pos1][pos2].grid()
        ax[pos1][pos2].legend()


    plt.suptitle(f'Cumulative Train Results - {strategy_name}', fontsize = 30)
    plt.savefig(f'results/{ts_dir}/{dataset_name}/{cicle_iter}/{strategy_name}/train_val_plots/cumulative_train_results.png')



def write_csv(task: str, ts_dir: str, dataset_name: str, head: List[str], values: List[Any]) -> None:
    res_path = os.path.join('results', ts_dir, dataset_name, f'{task}_results.csv')
    
    if (not os.path.exists(res_path)):
        
        with open(res_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            f.close()
    
    with open(res_path, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(values)
        f.close()
                 
            

def create_directory(dir: str) -> None: os.makedirs(dir, exist_ok=True)

def create_ts_dir(timestamp: str, dataset_name: str, iter: str) -> None:
    create_directory(os.path.join('results', timestamp, dataset_name, iter))
    
def create_method_res_dir(path: str) -> None:
    for dir in ['train_val_plots', 'tsne_plots', 'new_labeled_images']:
        create_directory(os.path.join(path, dir))
    
def create_class_dir(base_path: str, iter: int, classes: List[str]) -> None:
    for cls in classes: create_directory(f'{base_path}/new_labeled_images/{iter}/{cls}')
        
    
def get_palette(n_classes: int) -> List[ColorType]:
    if n_classes <= 10: palette = list(mcolors.TABLEAU_COLORS.values())
    elif n_classes <= 100: palette = list(mcolors.CSS4_COLORS.values())
    else: palette = list(mcolors.XKCD_COLORS.values())
    random.shuffle(palette)
    return palette
    

def plot_gtg_entropy_tensor(tensor: torch.Tensor, topk: List[int], lab_unlabels: List[int], \
                           classes: List[str], path: str, iter: int, max_x: int, dir: str) -> None:
    
    create_directory(f'{path}/gtg_entropies_plots/{dir}')
    
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    
    x = np.arange(max_x)
    array = tensor.cpu().numpy()
    
    new_labeled = array[topk]
    new_labeled_lab = [lab_unlabels[k] for k in topk]
    
    mask = np.ones(array.shape[0], dtype=bool)
    mask[topk] = False
    unlabeled = array[mask]

    title = 'Entropy History' if dir == 'history' else 'Entropy Derivatives'

    palette = get_palette(len(classes))
        
    pl_cls_1, pl2_cls_2 = set(), set()

    for i, lab in zip(range(len(array)), lab_unlabels):
        if lab not in pl_cls_1 and len(classes) <= 10 :
            axes[0].plot(x, array[i], linestyle="-", label=classes[lab], color=palette[lab])
            pl_cls_1.add(lab)
        else:
            axes[0].plot(x, array[i], linestyle="-", color=palette[lab])
    
    axes[0].set_title(f'{title} Classes', fontsize = 15)
    axes[0].set_ylabel('Entropy', fontsize = 10)
    axes[0].set_xlabel('GTG Iterations', fontsize = 10)
    axes[0].grid()
    axes[0].legend()
    
    
    for array, color, style, label in [(unlabeled, 'lightblue', '--', 'unlabeled'),
                                       (new_labeled, None, '-', new_labeled_lab)]:
        if style == '-':
            labels_array = label
        
        for i in range(len(array)):
            
            if style == '-':
                color = palette[labels_array[i]]
                label = classes[labels_array[i]]
                        
            if label not in pl2_cls_2 and len(classes) <= 10:
                axes[1].plot(x, array[i], linestyle=style, label=label, color=color)
                pl2_cls_2.add(label)
            else: axes[1].plot(x, array[i], linestyle=style, color=color)
        
    axes[1].set_ylabel('Entropy', fontsize = 10)
    axes[1].set_xlabel('GTG Iterations', fontsize = 10)
    axes[1].set_title(f'{title} - New_Lab / Unlab', fontsize = 15)
    axes[1].grid()
    axes[1].legend()
    
    plt.suptitle(f'Entropy History - Iteration {iter}', fontsize = 30)
    plt.savefig(f'{path}/gtg_entropies_plots/{dir}/{iter}.png')
    



def plot_res_std_mean(task: str, timestamp: str, dataset_name: str) -> None:
    df = pd.read_csv(f'results/{timestamp}/{dataset_name}/{task}_results.csv')

    df_grouped = (
        df[
            ['method', 'lab_obs', 'test_accuracy' if task == 'clf' else 'test_map']
        ].groupby(['method', 'lab_obs']).agg(['mean', 'std', 'count'])
    )
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    
    df_grouped['ci'] = 1.96 * df_grouped['std'] / np.sqrt(df_grouped['count'])
    df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
    df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
    
    methods = df_grouped['method'].unique()
    shapes = ['o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v']

    lines_handles = []
    palette = get_palette(len(methods))

    # Plot each method
    plt.figure(figsize=(14, 10))
    for idx, method in enumerate(methods):
        method_data = df_grouped[df_grouped['method'] == method]
        plt.plot(method_data['lab_obs'], method_data['mean'], label=method, 
                 linestyle = 'dashed' if method.split('_')[1] in 'Random' else 'solid', color=palette[idx])
        plt.fill_between(method_data['lab_obs'], method_data['ci_lower'], method_data['ci_upper'], alpha=0.3, color=palette[idx])
        plt.scatter(method_data['lab_obs'], method_data['mean'], marker=shapes[idx], color=palette[idx], zorder=5)
        lines_handles.append(mlines.Line2D([], [], color=palette[idx], marker=shapes[idx], markersize=5, label=method))
        

    plt.xlabel('Labeled Observations', fontsize = 15)
    plt.ylabel('Test Accuracy' if task == 'clf' else 'Test mAP', fontsize = 15)
    plt.title(f'{dataset_name} results', fontsize = 30)
    plt.legend(handles=lines_handles)
    plt.grid(True)
    
    plt.savefig(f'results/{timestamp}/{dataset_name}/mean_std_results.png') 
    


def set_seeds(seed: int = 10001) -> None:
    # setting seed and deterministic behaviour of pytorch for reproducibility
    # https://discuss.pytorch.org/t/determinism-in-pytorch-across-multiple-files/156269
    #os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def plot_tsne_A(A: Tuple[torch.Tensor, torch.Tensor], labels: Tuple[torch.Tensor, torch.Tensor],
                classes: List[str], time_stamp: str, ds_name: str, samp_iter: int, method: str,
                affinity: str, strategy: str, iter: int) -> None:
    
    A_1, A_2 = A
    
    label_lab, unlabel_lab = labels
    len_lab = len(label_lab)
    
    tsne_A1 = TSNE().fit_transform(A_1.cpu().numpy())
    tsne_A2 = TSNE().fit_transform(A_2.cpu().numpy())
    
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(23, 18))
        
    for idx, (tsne, name) in enumerate(zip([tsne_A1, tsne_A2], ['Original', f'{strategy} Sparsification'])):
        
        tsne_lab, tsne_unlab = tsne[:len_lab,:], tsne[len_lab:,:]
        x_lab, y_lab = tsne_lab[:,0], tsne_lab[:,1]
        x_unlab, y_unlab = tsne_unlab[:,0], tsne_unlab[:,1]
        x, y = np.hstack((x_lab, x_unlab)), np.hstack((y_lab, y_unlab))
        label = np.hstack((label_lab, unlabel_lab))
        
        sns.scatterplot(x=x_unlab, y=y_unlab, label='unlabeled', color='blue', s=17, ax=axes[idx][0])
        sns.scatterplot(x=x_lab, y=y_lab, label='labeled', color='orange', s=17, ax=axes[idx][0])
        axes[idx][0].set_title(f'{name} Affinity Matrix', fontsize = 15)
        axes[idx][0].legend()
        
        sns_p = sns.scatterplot(x=x, y=y, hue=[classes[l] for l in label], s=17, ax=axes[idx][1])
        axes[idx][1].set_title(f'{name} Affnity Matrix Classes', fontsize = 15)
        axes[idx][1].legend()
        if len(classes) > 10: sns_p.legend([],[], frameon=False)

    plt.suptitle(f'Affinity Matrix TSNE Plots - Iteration {iter}', fontsize=30)
    
    create_directory(f'results/{time_stamp}/{ds_name}/{samp_iter}/{method}/tsne_plots/{affinity}_matrix')
    plt.savefig(f'results/{time_stamp}/{ds_name}/{samp_iter}/{method}/tsne_plots/{affinity}_matrix/{iter}.png')
    
    
    
def plot_new_labeled_tsne(lab: Dict[str, torch.Tensor], unlab: Dict[str, torch.Tensor],
                          iter: str, method: str, ds_name: str, idxs_new_labels: List[int], 
                          classes: List[str], time_stamp: str, samp_iter: int, d_labels: Dict[str, int],
                          gtg_result_prediction = None):
    
    tsne = TSNE().fit_transform(np.vstack((lab['embedds'].cpu().numpy(), unlab['embedds'].cpu().numpy())))
    tsne_lab, tsne_unlab = tsne[:len(lab['embedds']), :], tsne[len(lab['embedds']):, :]
    
    x_lab, y_lab = tsne_lab[:, 0], tsne_lab[:, 1]
    x_unlab, y_unlab = tsne_unlab[:, 0], tsne_unlab[:, 1]
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 14))
    
    if isinstance(gtg_result_prediction, np.ndarray):
        
        idxs_unlab_again = [id for id in range(len(gtg_result_prediction)) if id not in idxs_new_labels]
        
        pred_unlab = gtg_result_prediction[idxs_unlab_again]
        pred_new_lab = gtg_result_prediction[idxs_new_labels]
        
        sns.scatterplot(x=x_unlab[idxs_unlab_again], y=y_unlab[idxs_unlab_again], 
                        label='unlabeled', color='blue', s=17, ax=axes[0], style=pred_unlab, 
                        markers=['X', 'o'] if len(np.unique(pred_unlab)) == 2 else ['o'])
        
        sns.scatterplot(x=x_lab, y=y_lab, label='labeled', color='orange', s=17, ax=axes[0], markers=['o'])
        
        sns.scatterplot(x=x_unlab[idxs_new_labels], y=y_unlab[idxs_new_labels], label='new_labeled', 
                        s=17, color='red', ax=axes[0], style=pred_new_lab, 
                        markers=['X', 'o'] if len(np.unique(pred_new_lab)) == 2 else ['o'])
    else:
        
        sns.scatterplot(x=x_unlab, y=y_unlab, label='unlabeled', color='blue', s=17, ax=axes[0], markers=['o'])
        sns.scatterplot(x=x_lab, y=y_lab, label='labeled', color='orange', s=17, ax=axes[0], markers=['o'])
        sns.scatterplot(x=x_unlab[idxs_new_labels], y=y_unlab[idxs_new_labels], label='new_labeled', 
                        s=17, color='red', ax=axes[0], markers=['o'])
    
    axes[0].set_title('TSNE -- labeled - unlabeled - new_labeled', fontsize = 15)
    axes[0].legend()
    
    sns_p = sns.scatterplot(x=np.hstack((x_lab, x_unlab)), y=np.hstack((y_lab, y_unlab)), 
                    hue=[classes[l] for l in np.hstack((lab['labels'], unlab['labels']))], s=17, ax=axes[1])
        
    axes[1].set_title('TSNE - classes', fontsize = 15)
    axes[1].legend()    
    if len(classes) <= 10: 
        fig.text(0.5, 0.05, f'New Labeled Observations: {d_labels}', ha='center', va='center', fontdict={'size':17})
    else: sns_p.legend([],[], frameon=False)
    
    plt.suptitle(f'{ds_name} - {method} - {iter}', fontsize = 30)
    plt.savefig(f'results/{time_stamp}/{ds_name}/{samp_iter}/{method}/tsne_plots/{iter}.png')

    
    
    
def count_class_observation(classes: List[str], dataset: Dataset, topk_idx_obs=None) -> Dict[str, int]:
    if topk_idx_obs == None: labels = [lab for _, _ ,lab in dataset] if len(dataset[0]) == 3 else [lab for _, _ ,lab, _ in dataset] # -> for TiDAL
    else: labels = [dataset[k][2] for k in topk_idx_obs]
    
    d_labels = {}
    for cls in classes: d_labels[cls] = 0
    keys_d_labels = list(d_labels.keys())
    for l in labels: d_labels[keys_d_labels[l]] += 1
    return d_labels


def cycle(iterable):
    while True:
        for x in iterable: yield x
        
           
# weights initiaization
def init_weights_apply(m: torch.nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=1e-3)
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def log_assert(condition: bool | torch.Tensor, message: str):
    try: assert condition
    except AssertionError as err:
        logger.exception(message)
        raise err

    
'''def download_coco_dataset() -> None:
    if not os.path.exists('datasets/coco'):
        logger.info(' => Downloading COCO-2017 Dataset')
                
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
        logger.info('COCO-2017 Dataset already downloaded')'''