
import torch
import torch.nn as nn
import torch.nn.init as init

import matplotlib.pyplot as plt

import csv
import os
import errno

from ResNet18 import BasicBlock

def save_init_checkpoint(model, optimizer, scheduler):
    checkpoint = { 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict() }
    torch.save(checkpoint, 'app/checkpoints/init_checkpoint.pth.tar')


def get_mean_std(dataloader):
    mean = 0.0
    std = 0.0

    for _, images, _ in dataloader:
        # Assuming images are in the shape (batch_size, channels, height, width)
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)
    
    return mean, std
    

def accuracy_score(output, label):
    output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
    return (output_class == label).sum().item()/len(output)



def entropy(tensor):
    #x = torch.clone(tensor) + 1e-20
    x = tensor + 1e-20
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
    
    if save_plot: plt.savefig(f'results/{ts_dir}/{plot_png_name}')
    else: plt.show()
    
    
    
def save_train_val_curves(results_info, ts_dir, al_iter):

    res = results_info['results']
    epochs = range(1, len(res['train_loss']) + 1)

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18,8))
    minloss_val = min(res['val_loss'])
    minloss_ep = res['val_loss'].index(minloss_val) + 1

    maxacc_ep = res['val_accuracy'][minloss_ep - 1]
    

    ax[0].plot(epochs, res['train_loss'], label = 'train_loss')
    ax[0].plot(epochs, res['val_loss'], label = 'val_loss')
    ax[0].axvline(minloss_ep, linestyle='--', color='r', label='Early Stopping Checkpoint')

    ax[0].axhline(minloss_val, linestyle='--', color='r')
    ax[0].set_title('Loss - Epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].grid()
    ax[0].legend()


    ax[1].plot(epochs, res['train_accuracy'], label = 'train_accuracy_score')
    ax[1].plot(epochs, res['val_accuracy'], label = 'val_accuracy_score')
    ax[1].axvline(minloss_ep, linestyle='--', color='r', label='Early Stopping Checkpoint')

    ax[1].axhline(maxacc_ep, linestyle='--', color='r')
    ax[1].set_title('Accuracy Score - Epochs')
    ax[1].set_ylabel('Accuracy Score')
    ax[1].set_xlabel('Epochs')
    ax[1].grid()
    ax[1].legend()
    plt.suptitle(f'AL iter {al_iter} - {results_info["model_name"]}', fontsize = 30)
    
    plt.savefig(f'results/{ts_dir}/train_val_plots/{al_iter}_{results_info["model_name"]}.png')



def write_csv(ts_dir, head, values):
    if (not os.path.exists(f'results/{ts_dir}/results.csv')):
        
        with open(f'results/{ts_dir}/results.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            f.close()
    
    with open(f'results/{ts_dir}/results.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(values)
        f.close()
                 
            
            
def create_ts_dir_res(timestamp):
    mydir = os.path.join('results', timestamp)
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    mydir = os.path.join('results', timestamp, 'train_val_plots')
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        
        
        
        
# weights initiaization
'''def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)'''
                
def init_params(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=1e-3)
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, BasicBlock):
        for c in list(m.children()): init_params(c)