
import torch
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data.sampler import SubsetRandomSampler
import torch.distributed as dist

from DDPTrainer import TrainDDP
from ResNet18 import BasicBlock, ResNet_Weird
from Datasets import DatasetChoice, SubsetDataloaders
from utils import save_train_val_curves, write_csv, set_seeds

import copy
import random
import os



class TrainEvaluate(object):

    def __init__(self, params, LL):

        self.LL = LL        
        sdl: SubsetDataloaders = params['DatasetChoice']
        
        self.n_classes = sdl.n_classes
        self.n_channels = sdl.n_channels
        self.dataset_id = sdl.dataset_id
        
        
        self.test_ds: DatasetChoice = sdl.test_ds
        self.val_ds: DatasetChoice = sdl.val_ds
        
        self.transformed_trainset: DatasetChoice = sdl.transformed_trainset 
        self.non_transformed_trainset: DatasetChoice = sdl.non_transformed_trainset 
        
        self.device = params['device']
        self.batch_size = params['batch_size']
       
        self.patience = params['patience']
        self.score_fn = params['score_fn']
        self.timestamp = params['timestamp']
        self.dataset_name = params['dataset_name']
        self.iter_sample = params['samp_iter']

        self.best_check_filename = f'app/checkpoints/{self.dataset_name}'
                
        # I need the deep copy only of the list of labeled and unlabeled indices
        self.labeled_indices = copy.deepcopy(sdl.labeled_indices)
        self.unlabeled_indices = copy.deepcopy(sdl.unlabeled_indices)
        
        
        ###########
        set_seeds()
        ###########
        
        
        self.lab_train_dl = DataLoader(
            self.transformed_trainset, batch_size=self.batch_size, 
            sampler=SubsetRandomSampler(self.labeled_indices), pin_memory=True
        )
        
        self.model = ResNet_Weird(BasicBlock, [2, 2, 2, 2], num_classes=self.n_classes, n_channels=self.n_channels).to(self.device)
    

        
        
    def get_embeddings(self, dataloader, dict_to_modify):
        
        checkpoint = torch.load(f'{self.best_check_filename}/best_{self.method_name}.pth.tar', map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

        if 'embedds' in dict_to_modify:
            dict_to_modify['embedds'] = torch.empty((0, self.model.linear.in_features), dtype=torch.float32, device=self.device)
        if 'probs' in dict_to_modify:
            dict_to_modify['probs'] = torch.empty((0, self.n_classes), dtype=torch.float32, device=self.device)
        if 'labels' in dict_to_modify:
            dict_to_modify['labels'] = torch.empty(0, dtype=torch.int8, device=self.device)
        if 'idxs' in dict_to_modify:
            dict_to_modify['idxs'] = torch.empty(0, dtype=torch.int8, device=self.device)
        
        self.model.eval()

        # again no gradients needed
        with torch.inference_mode():
            for idxs, images, labels in dataloader:
                
                idxs, images, labels = idxs.to(self.device), images.to(self.device), labels.to(self.device)
                outs, embed, _, _ = self.model(images)
                
                if 'embedds' in dict_to_modify:
                    dict_to_modify['embedds'] = torch.cat((dict_to_modify['embedds'], embed.squeeze()), dim=0)
                if 'probs' in dict_to_modify:
                    dict_to_modify['probs'] = torch.cat((dict_to_modify['probs'], outs.squeeze()), dim=0)
                if 'labels' in dict_to_modify:
                    dict_to_modify['labels'] = torch.cat((dict_to_modify['labels'], labels), dim=0)
                if 'idxs' in dict_to_modify:
                    dict_to_modify['idxs'] = torch.cat((dict_to_modify['idxs'], idxs), dim=0)    



    def get_new_dataloaders(self, overall_topk):
        
        # extend with the overall_topk
        self.labeled_indices.extend(overall_topk)
        
        # remove new labeled observations
        for idx_to_remove in overall_topk: self.unlabeled_indices.remove(idx_to_remove)
        
        # sanity check
        if len(list(set(self.unlabeled_indices) & set(self.labeled_indices))) == 0:
            print('Intersection between indices is EMPTY')
        else: raise Exception('NON EMPTY INDICES INTERSECTION')


        # generate the new labeled DataLoader
        self.lab_train_dl = DataLoader(
            self.transformed_trainset, batch_size=self.batch_size, 
            sampler=SubsetRandomSampler(self.labeled_indices),
            pin_memory=True
        )




    def get_unlabebled_samples(self, unlab_sample_dim, iter):
        if(len(self.unlabeled_indices) > unlab_sample_dim):
            
            # custom seed for the sequence
            random.seed(iter * self.dataset_id + 100 * self.iter_sample)
            
            seq = random.sample(self.unlabeled_indices, unlab_sample_dim)
            #printing only the last 5 sampled unlabeled observations
            print(seq[-5:])
            
            # set back the seed to the initial one, the one set on the main file
            random.seed(100001)
            
            return seq
        
        else: return self.unlabeled_indices
        
    
    
    def remove_model_opt(self):
        del self.model
        torch.cuda.empty_cache()
        
        
        
    def clear_cuda_variables(self, variables):
        for var in variables: del var
        torch.cuda.empty_cache()




    def train_evaluate_save(self, epochs, lab_obs, iter, results):
                
        params = {
            'num_classes': self.n_classes, 'n_channels': self.n_channels,
            'LL': self.LL, 'patience': self.patience, 'dataset_name': self.dataset_name, 'method_name': self.method_name,
            'train_ds': Subset(self.transformed_trainset, self.labeled_indices), 'val_ds': self.val_ds, 'test_ds': self.test_ds,
            'transformed_trainset': self.transformed_trainset, 'non_transformed_trainset': self.non_transformed_trainset,
            'batch_size': self.batch_size, 'score_fn': self.score_fn
        }
        
        
        
        
        parent_conn, child_conn = mp.Pipe()
        world_size = torch.cuda.device_count()
        mp.spawn(train_ddp, args=(world_size, params, epochs, child_conn, ), nprocs=world_size)
        while parent_conn.poll():
            train_recv, test_recv = parent_conn.recv()
                
    
            
        
        train_results = {
            'model_name': self.method_name, 
            'results': {
                    'train_loss': train_recv[0], 'train_loss_ce': train_recv[1],
                    'train_loss_weird': train_recv[2], 'train_accuracy': train_recv[3], 
                    'val_loss': train_recv[4], 'val_loss_ce': train_recv[5],
                    'val_loss_weird': train_recv[6], 'val_accuracy': train_recv[7]
                }
        }
             
        test_accuracy, test_loss, test_loss_ce, test_loss_weird = test_recv
             
        results['test_accuracy'].append(test_accuracy)
        results['test_loss'].append(test_loss)
        results['test_loss_ce'].append(test_loss_ce)
        results['test_loss_weird'].append(test_loss_weird)

        write_csv(
            ts_dir = self.timestamp,
            dataset_name = self.dataset_name,
            head = ['method', 'iter', 'lab_obs', 'test_accuracy', 'test_loss', 'test_loss_ce', 'test_loss_weird'],
            values = [self.method_name, self.iter_sample, lab_obs, test_accuracy, test_loss, test_loss_ce, test_loss_weird]
        )
        
        save_train_val_curves(train_results, self.timestamp, self.dataset_name, iter, self.iter_sample, self.LL)

        
        
def train_ddp(rank, world_size, params, epochs, conn):
    
    #MASTER_ADDR:MASTER_PORT=ppv-gpu1:16217
    os.environ["MASTER_ADDR"] = "ppv-gpu1"
    os.environ["MASTER_PORT"] = "16217"
    
    init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    
    
    
    train_results = [torch.zeros((8, epochs), device=rank) for _ in range(world_size)]
    test_results = [torch.zeros(4, device=rank) for _ in range(world_size)]
    
    
    
    batch_size = int(params['batch_size'])

    model = ResNet_Weird(BasicBlock, [2, 2, 2, 2], num_classes=params['num_classes'], n_channels=params['n_channels']).cuda(rank)#to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    params['model'] = model
    
    num_workers = int((int(os.environ['SLURM_CPUS_PER_TASK']) + world_size - 1) / world_size)
    
    
    params['train_dl'] = DataLoader(
                            params['train_ds'], batch_size=batch_size,
                            sampler=DistributedSampler(params['train_ds'], num_replicas=world_size, rank=rank, shuffle=True, seed=100001),
                            shuffle=False, pin_memory=True, persistent_workers=True,
                            num_workers=num_workers
                        )
    
    params['val_dl'] = DataLoader(
                            params['val_ds'], batch_size=batch_size,
                            sampler=DistributedSampler(params['val_ds'], num_replicas=world_size, rank=rank, shuffle=False, seed=100001),
                            shuffle=False, pin_memory=True, persistent_workers=True,
                            num_workers=num_workers
                        )
    
    params['test_dl'] = DataLoader(
                            params['test_ds'], batch_size=batch_size,
                            sampler=DistributedSampler(params['test_ds'], num_replicas=world_size, rank=rank, shuffle=False, seed=100001),
                            shuffle=False, pin_memory=True, persistent_workers=True,
                            num_workers=num_workers
                        )
   
    
    train_test = TrainDDP(rank, params)
    
    
    if rank == 0:
        dist.gather(train_test.train_evaluate(epochs), train_results)
        dist.gather(train_test.test(), test_results)
    else:
        dist.gather(train_test.train_evaluate(epochs))
        dist.gather(train_test.test())
                
        
    
    if rank == 0:
        train_results = (torch.sum(torch.stack(train_results), dim=0) / world_size).cpu().tolist()
        test_results = (torch.sum(torch.stack(test_results), dim=0) / world_size).cpu().tolist()
        
        conn.send((train_results, test_results))           
    
    
    destroy_process_group()
    
    