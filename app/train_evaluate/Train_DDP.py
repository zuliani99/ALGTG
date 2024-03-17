
import torch

from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from ResNet18 import BasicBlock, ResNet_Weird
from train_evaluate.TrainWorker import TrainWorker

import os
from typing import Tuple, Dict, Any, List
from multiprocessing import connection 


def cleanup() -> None:
    dist.destroy_process_group()
    
    
def initialize_preocess(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "ppv-gpu1"
    os.environ["MASTER_PORT"] = "16217"
    
    init_process_group(backend='nccl', init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}', rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    


def train_ddp(rank: int, world_size: int, params: Dict[str, Any], epochs: int, conn: connection.Connection) -> None:
    
    initialize_preocess(rank, world_size)
    
    train_results = [torch.zeros((8, epochs), device=rank) for _ in range(world_size)]
    test_results = [torch.zeros(4, device=rank) for _ in range(world_size)]
    
    
    model = ResNet_Weird(BasicBlock, [2, 2, 2, 2], image_size=params['image_size'], num_classes=params['num_classes'], n_channels=params['n_channels']).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    params['model'] = model
    
    num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    
    
    params['train_dl'] = DataLoader(
                            params['train_ds'], batch_size=params['batch_size'],# // world_size,
                            sampler=DistributedSampler(params['train_ds'], num_replicas=world_size, rank=rank, shuffle=True, seed=100001),
                            shuffle=False, pin_memory=True, persistent_workers=True,
                            num_workers=num_workers
                        )
    
    params['val_dl'] = DataLoader(
                            params['val_ds'], batch_size=params['batch_size'],# // world_size,
                            sampler=DistributedSampler(params['val_ds'], num_replicas=world_size, rank=rank, shuffle=False, seed=100001),
                            shuffle=False, pin_memory=True, persistent_workers=True,
                            num_workers=num_workers
                        )
    
    params['test_dl'] = DataLoader(
                            params['test_ds'], batch_size=params['batch_size'],# // world_size,
                            sampler=DistributedSampler(params['test_ds'], num_replicas=world_size, rank=rank, shuffle=False, seed=100001),
                            shuffle=False, pin_memory=True, persistent_workers=True,
                            num_workers=num_workers
                        )
   
    
    train_test = TrainWorker(rank, params, world_size)
    
    
    if rank == 0:
        dist.gather(train_test.train_evaluate(epochs), train_results)
        dist.gather(train_test.test(), test_results)
    else:
        dist.gather(train_test.train_evaluate(epochs))
        dist.gather(train_test.test())
              
    dist.barrier()
    
    if rank == 0:
        train_results = (torch.sum(torch.stack(train_results), dim=0) / world_size).cpu().tolist()
        test_results = (torch.sum(torch.stack(test_results), dim=0) / world_size).cpu().tolist()
        
        conn.send((train_results, test_results))           
    
    ##################
    # see if this remove the warning of multiprocessing
    dist.barrier()
    ##################
    
    cleanup()
    
    
    
def train(params: Dict[str, Any], epochs: int) -> Tuple[List[float], List[float]]:
    
    params['model'] = ResNet_Weird(BasicBlock, [2, 2, 2, 2], image_size=params['image_size'], num_classes=params['num_classes'], n_channels=params['n_channels']).to(params['main_device'])
   
    params['val_dl'] = DataLoader(params['val_ds'], batch_size=params['batch_size'], shuffle=False, pin_memory=True)
    params['test_dl'] = DataLoader(params['test_ds'], batch_size=params['batch_size'], shuffle=False, pin_memory=True)
    
    train_test = TrainWorker(params['main_device'], params)           
        
    train_results = train_test.train_evaluate(epochs).cpu().tolist()
    test_results = train_test.test().cpu().tolist()
    
    return train_results, test_results