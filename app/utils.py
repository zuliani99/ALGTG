
import torch

import torch.nn as nn
import torch.nn.init as init

import matplotlib.pyplot as plt

import csv
import os
import errno

from ResNet18 import BasicBlock


def save_init_checkpoint(model, optimizer, scheduler):
    print(' => Saving initial checkpoint')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(checkpoint, 'app/checkpoints/init_checkpoint.pth.tar')
    print(' DONE\n')


def get_mean_std(dataloader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for _, images, _ in dataloader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std

    

def accuracy_score(output, label):
    output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
    return (output_class == label).sum().item()/len(output)



def entropy(tensor):
    x = tensor + 1e-20
    return -torch.sum(x * torch.log2(x), dim=1)

    

def plot_loss_curves(methods_results, n_lab_obs, save_plot, ts_dir, plot_png_name = None):
    
    _, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (28,18))
    
    # test_loss
    for method_str, values in methods_results.items():
        if(isinstance(list(values.keys())[0], int)):
            for n_samples, results in values.items():
                ax[0][0].plot(n_lab_obs, results['test_loss'], label = f'{method_str} - {str(n_samples)} splits')
        else:
            ax[0][0].plot(n_lab_obs, values['test_loss'], label = f'{method_str}')

    ax[0][0].set_title('Total Loss - # Labeled Obs')
    ax[0][0].set_xlabel('# Labeled Obs')
    ax[0][0].set_ylabel('Total Loss')
    ax[0][0].grid()
    ax[0][0].legend()

        
    # test_accuracy
    for method_str, values in methods_results.items():
        if(isinstance(list(values.keys())[0], int)):
            for n_samples, results in values.items():
                ax[0][1].plot(n_lab_obs, results['test_accuracy'], label = f'{method_str} - {str(n_samples)} splits')
        else:
            ax[0][1].plot(n_lab_obs, values['test_accuracy'], label = f'{method_str}')
                
    ax[0][1].set_title('Accuracy Score - # Labeled Obs')
    ax[0][1].set_ylabel('Accuracy Score')
    ax[0][1].set_xlabel('# Labeled Obs')
    ax[0][1].grid()
    ax[0][1].legend()
        
        
    # test_loss_ce
    for method_str, values in methods_results.items():
        if(isinstance(list(values.keys())[0], int)):
            for n_samples, results in values.items():
                ax[1][0].plot(n_lab_obs, results['test_loss_ce'], label = f'{method_str} - {str(n_samples)} splits')
        else:
            ax[1][0].plot(n_lab_obs, values['test_loss_ce'], label = f'{method_str}')
                
    ax[1][0].set_title('CE Loss - # Labeled Obs')
    ax[1][0].set_ylabel('CE Loss')
    ax[1][0].set_xlabel('# Labeled Obs')
    ax[1][0].grid()
    ax[1][0].legend()
        
        
    # test_loss_weird
    for method_str, values in methods_results.items():
        if(isinstance(list(values.keys())[0], int)):
            for n_samples, results in values.items():
                ax[1][1].plot(n_lab_obs, results['test_loss_weird'], label = f'{method_str} - {str(n_samples)} splits')
        else:
            ax[1][1].plot(n_lab_obs, values['test_loss_weird'], label = f'{method_str}')
                
    ax[1][1].set_title('Loss Weird - # Labeled Obs')
    ax[1][1].set_ylabel('Loss Weird')
    ax[1][1].set_xlabel('# Labeled Obs')
    ax[1][1].grid()
    ax[1][1].legend()
    

    plt.suptitle('Results', fontsize = 30)
    
    if save_plot: plt.savefig(f'results/{ts_dir}/{plot_png_name}')
    else: plt.show()
    
    
    
def save_train_val_curves(results_info, ts_dir, al_iter, flag_LL):

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

    plt.suptitle(f'AL iter {al_iter} - {results_info["model_name"]}', fontsize = 30)


    if(not os.path.exists(f'results/{ts_dir}/train_val_plots/{results_info["model_name"]}')):
        os.makedirs(f'results/{ts_dir}/train_val_plots/{results_info["model_name"]}')
    plt.savefig(f'results/{ts_dir}/train_val_plots/{results_info["model_name"]}/{al_iter}.png')



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
        
        