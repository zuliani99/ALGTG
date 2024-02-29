
import torch

import torch.nn as nn
import torch.nn.init as init

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import csv
import os

    

def accuracy_score(output, label):
    output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
    return (output_class == label).sum().item()/len(output)


def entropy(tensor, dim=1):
    x = tensor + 1e-20
    return -torch.sum(x * torch.log2(x), dim=dim)

    

def plot_loss_curves(methods_results, n_lab_obs, ts_dir, save_plot=True, plot_png_name=None):
    
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
    
    
    
def save_train_val_curves(results_info, ts_dir, dataset_name, al_iter, cicle_iter, flag_LL):

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



def write_csv(ts_dir, dataset_name, head, values):
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
                 
            

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

            
def create_ts_dir_res(timestamp, dataset_name, iter):
    mydir = os.path.join('results', timestamp)
    create_directory(mydir)
    create_directory(os.path.join(mydir, dataset_name))
    create_directory(os.path.join(mydir, dataset_name, iter))
    create_directory(os.path.join(mydir, dataset_name, iter, 'train_val_plots'))

        
        
# weights initiaization
def init_weights_apply(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=1e-3)
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
        
        
        
def plot_story_tensor(story_tensor, path, iter, max_x):
    import numpy as np
    
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



def plot_accuracy_std_mean(timestamp, dataset_name):
    df = pd.read_csv(f'results/{timestamp}/{dataset_name}/results.csv')

    df_grouped = (
        df[['method', 'lab_obs', 'test_accuracy']].groupby(['method', 'lab_obs']).agg(['mean', 'std', 'count'])
    )
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    
    df_grouped['ci'] = 1.96 * df_grouped['std'] / np.sqrt(df_grouped['count'])
    df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
    df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
    
    methods = df_grouped['method'].unique()

    # Plot each method
    plt.figure(figsize=(14, 10))
    for method in methods:
        method_data = df_grouped[df_grouped['method'] == method]
        plt.plot(method_data['lab_obs'], method_data['mean'], label=method)
        plt.fill_between(method_data['lab_obs'], method_data['ci_lower'], method_data['ci_upper'], alpha=0.3)

    plt.xlabel('Labeled Observations')
    plt.ylabel('Test Accuracy')
    plt.title(f'{dataset_name} results')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'results/{timestamp}/{dataset_name}/mean_std_accuracy_plot.png') 
 