import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

import csv
import os
import errno


def get_initial_dataloaders(trainset, val_rateo, labeled_ratio, batch_size):

    train_size = len(trainset) #50000

    val_size = int(train_size * val_rateo)
    train_size -= val_size

    train_data, val_data = random_split(trainset, [int(train_size), int(val_size)])

    # validation dataloader
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    #-------------

    train_data_size = len(train_data)

    # Calculate the number of samples for each split
    labeled_size = int(labeled_ratio * train_data_size)
    unlabeled_size = train_data_size - labeled_size

    # Get the dataset split
    labeled_set, unlabeled_set = random_split(train_data, [labeled_size, unlabeled_size])

    # Obtain the splitted dataloader
    labeled_train_dl = DataLoader(labeled_set, batch_size=batch_size, shuffle=True, num_workers=2)
    unlabeled_test_dl = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return (labeled_train_dl, unlabeled_test_dl), (labeled_set, unlabeled_set), train_dl, val_dl



def get_overall_top_k(topk_list, top_k):

    # Concatenate all the top_k.values tensors
    all_values = torch.cat([topk_result.values for topk_result in topk_list])

    # Find the overall top k values and their corresponding indices
    overall_topk_values, overall_topk_indices = torch.topk(all_values, k=top_k)

    overall_top_k = []

    # Print the results
    for i in range(len(overall_topk_values)):
        index_position = overall_topk_indices[i].item()

        # Find the list index and index value within the specific top_k result
        list_index = 0
        current_values_count = 0
        for topk_result in topk_list:
            if index_position < current_values_count + len(topk_result.values):
                index_value = index_position - current_values_count
                break

            current_values_count += len(topk_result.values)
            list_index += 1

        overall_top_k.append((list_index, index_value))

    return overall_top_k



def accuracy_score(output, label):
    output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
    return (output_class == label).sum().item()/len(output)



def get_resnet18(n_classes):
    resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='DEFAULT')

    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, n_classes)

    return resnet18

    

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



'''def delete_previous_csv():
            
    files = os.listdir('../temp_results') # 'temp_results'

    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join('../temp_results', file) # 'temp_results'
            os.remove(file_path)
'''            
            
            
def create_ts_dir_res(timestamp):
    mydir = os.path.join('../results', timestamp) #results
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..