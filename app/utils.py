import torch
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

import csv
import os
import errno
import copy


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
    labeled_train_dl = DataLoader(labeled_set, batch_size=batch_size, shuffle=True, num_workers=2)

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
        