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
    
    # train_data, val_data possono variare al massimo tra 0 -> 49999: 50000 osservazioni
    # train_data ha 40000 osservazioni 

    # validation dataloader
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)# True

    train_data_size = len(train_data)

    # Calculate the number of samples for each split
    labeled_size = int(labeled_ratio * train_data_size)
    unlabeled_size = train_data_size - labeled_size

    # Get the dataset split
    labeled_set, unlabeled_set = random_split(train_data, [labeled_size, unlabeled_size])
    # questi due hanno .indices basato su train_data
    
    # così io dovrei avere le referenze originali rispetto alla dimensione del traiset
    labeled_set_indices = [train_data.indices[id] for id in labeled_set.indices]
    unlabeled_set_indices = [train_data.indices[id] for id in unlabeled_set.indices]
    
    # labeled_set, unlabeled_set possono variare al massimo tra 0 -> 39999: 40000 osservazioni
    
    # Obtain the splitted dataloader
    labeled_train_dl = DataLoader(labeled_set, batch_size=batch_size, shuffle=False, num_workers=2)#True
    
    return labeled_train_dl, (labeled_set, unlabeled_set), val_dl, (labeled_set_indices, unlabeled_set_indices)



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

    _, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12,8))

    
    for method_str, values in methods_results.items():
        if(isinstance(list(values.keys())[0], int)):
            for n_samples, results in values.items():
                ax.plot(n_lab_obs, results['test_accuracy'], label = f'{method_str} - {str(n_samples)} splits')
        else:
            ax.plot(n_lab_obs, values['test_accuracy'], label = f'{method_str}')
            
    
    ax.set_title('Accuracy Score Results')
    ax.set_ylabel('Accuracy Score')
    ax.set_xlabel('# Labeled Obs')
    ax.grid()
    ax.legend()
    
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
        
        

'''def get_new_dataloaders(overall_topk, lab_train_ds, unlab_train_ds, train_ds, batch_size):

    new_lab_train_ds = np.array([
        np.array([
            #idx, image if isinstance(image, np.ndarray) else image.numpy(), label
            idx, image if isinstance(image, torch.Tensor) else torch.tensor(image), label
        ], dtype=object) for idx, image, label in tqdm(lab_train_ds, total=len(lab_train_ds), leave=False, desc='Copying lab_train_ds')], dtype=object)

    new_unlab_train_ds = np.array([
        np.array([
            #idx, image if isinstance(image, np.ndarray) else image.numpy(), label
            idx, image if isinstance(image, torch.Tensor) else torch.tensor(image), label
        ], dtype=object) for idx, image, label in tqdm(unlab_train_ds, total=len(unlab_train_ds), leave=False, desc='Copying unlab_train_ds')], dtype=object)
                

    for topk_idx in tqdm(overall_topk, total=len(overall_topk), leave=False, desc='Modifing the Unlabeled Dataset'):
            
        new_lab_train_ds = np.vstack((new_lab_train_ds, np.expand_dims(
            np.array([train_ds[topk_idx if isinstance(topk_idx, int) else topk_idx.item()][0],
                    train_ds[topk_idx if isinstance(topk_idx, int) else topk_idx.item()][1],
                    train_ds[topk_idx if isinstance(topk_idx, int) else topk_idx.item()][2]
            ], dtype=object)
        , axis=0)))
            
        for idx, (i, _, _) in enumerate(new_unlab_train_ds):
            if i == topk_idx if isinstance(topk_idx, int) else topk_idx.item():
                new_unlab_train_ds[idx] = np.array([np.nan, np.nan, np.nan], dtype=object)
            # set a [np.nan np.nan] the row and the get all the row not equal to [np.nan, np.nan]
        
        
    lab_train_ds = CIFAR10(None, new_lab_train_ds)
    unlab_train_ds = CIFAR10(None, new_unlab_train_ds[
        np.array(
            [not np.isnan(row[0])
                for row in tqdm(new_unlab_train_ds, total=len(new_unlab_train_ds),
                                leave=False, desc='Obtaining the unmarked observation from the Unlabeled Dataset')
            ]
        )])
        
    lab_train_dl = DataLoader(lab_train_ds, batch_size=batch_size, shuffle=True)
    
    return lab_train_ds, unlab_train_ds, lab_train_dl'''