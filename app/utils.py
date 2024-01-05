
import torch
import torch.nn as nn
import torch.nn.init as init

import matplotlib.pyplot as plt

import csv
import os
import errno

from ResNet18 import BasicBlock


def get_mean_std(dataloader):
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for _, image, _ in dataloader:
        channels_sum += torch.mean(image, dim=[0,2,3])
        channels_squared_sum += torch.mean(image**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    
    return mean, std
    
    

def accuracy_score(output, label):
    output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
    return (output_class == label).sum().item()/len(output)



def entropy(tensor):
    x = torch.clone(tensor) + 1e-20
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
    mydir = os.path.join('results', timestamp) #results
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        
        
        
# weights initiaization
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