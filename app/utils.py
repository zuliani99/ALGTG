
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.utils.data import DataLoader, Subset, random_split

import matplotlib.pyplot as plt

import csv
import os
import errno
import copy

from resnet.resnet_weird import BasicBlock


def get_initial_dataloaders(trainset, val_rateo, labeled_ratio, batch_size):

    train_size = len(trainset) #50000

    val_size = int(train_size * val_rateo)
    train_size -= val_size

    train_data, val_data = random_split(trainset, [int(train_size), int(val_size)])
    
    # train_data, val_data possono variare al massimo tra 0 -> 49999: 50000 osservazioni
    # train_data ha 40000 osservazioni 

    # validation dataloader
    #val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=1)# True
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    train_data_size = len(train_data)

    # Calculate the number of samples for each split
    labeled_size = int(labeled_ratio * train_data_size)
    unlabeled_size = train_data_size - labeled_size

    # Get the dataset split
    labeled_set, unlabeled_set = random_split(train_data, [labeled_size, unlabeled_size])
    # questi due hanno .indices basato su train_data
    
    # così io dovrei avere le referenze originali rispetto alla dimensione del traiset
    #labeled_set_indices = [train_data.indices[id] for id in labeled_set.indices]
    #unlabeled_set_indices = [train_data.indices[id] for id in unlabeled_set.indices]
    
    # labeled_set, unlabeled_set possono variare al massimo tra 0 -> 39999: 40000 osservazioni
    
    
    # SET THE INDICES OF THE LABELED AND UNLABELED SET CONSISTENT WRT THE ORIGINAL TRAINDATA

    labeled_set = Subset(trainset, [train_data.indices[id] for id in labeled_set.indices])
    unlabeled_set = Subset(trainset, [train_data.indices[id] for id in unlabeled_set.indices])
    
    # Obtain the splitted dataloader
    labeled_train_dl = DataLoader(labeled_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    
    return labeled_train_dl, (labeled_set, unlabeled_set), val_dl



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
    #print('ENTROPY OF TENSOR:   ', tensor)
    #x = copy.deepcopy(tensor.cpu()) + 1e-20
    #x = torch.clone(tensor).cpu() + 1e-20
    x = copy.deepcopy(tensor) + 1e-20
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
    ax[0].set_xticklabels(n_lab_obs, rotation=45)
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
    ax[1].set_xticklabels(n_lab_obs, rotation=45)
    ax[1].grid()
    ax[1].legend()

    plt.suptitle('Results', fontsize = 30)
    
    if save_plot: plt.savefig(f'results/{ts_dir}/{plot_png_name}') #results/{ts_dir}/{plot_png_name}
    else: plt.show()



def write_csv(ts_dir, head, values):
    if (not os.path.exists(f'results/{ts_dir}/results.csv')): #results/{ts_dir}/results.csv
        
        with open(f'results/{ts_dir}/results.csv', 'w', encoding='UTF8') as f: #results/{ts_dir}/results.csv
            writer = csv.writer(f)
            writer.writerow(head)
            f.close()
    
    with open(f'results/{ts_dir}/results.csv', 'a', encoding='UTF8') as f: #results/{ts_dir}/results.csv
        writer = csv.writer(f)
        writer.writerow(values)
        f.close()
                 
            
            
def create_ts_dir_res(timestamp):
    mydir = os.path.join('results', timestamp) #results
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        
        
        
# weights initiaization
'''def init_params(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias: init.xavier_uniform_(m.bias.data, gain=nn.init.calculate_gain('relu'))'''
def init_params(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, mode='fan_out')
        if m.bias is not None: init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight.data, std=1e-3)
        if m.bias is not None: init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight.data, 1)
        if m.bias is not None: init.constant_(m.bias.data, 0)
    elif isinstance(m, BasicBlock):
        for c in list(m.children()): init_params(c)

