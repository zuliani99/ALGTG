
import torch

from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from ..datasets_creation.Detection import detection_collate
from .Workers.Cls_TrainWorker import Cls_TrainWorker
from .Workers.Det_TrainWorker import Det_TrainWorker

import os
import gc
from typing import Tuple, Dict, Any, List
from multiprocessing import connection 


    
    
def initialize_preocess(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "ppv-gpu1"
    os.environ["MASTER_PORT"] = "16217"
    
    init_process_group(backend='nccl', init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}', rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    
    
    
def clear_process() -> None: destroy_process_group()



def train_ddp(rank: int, task: str, world_size: int, params: Dict[str, Any], conn: connection.Connection) -> None:
    
    initialize_preocess(rank, world_size)
    
    params['cp_t']['model'] = DDP(params['cp_t']['model'] , device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    
    train_ds = Subset(params['ct_p']['Dataset'].transformed_trainset, params['ct_p']['Dataset'].labeled_indices)
    
    if task == 'det': params['t_p']['epochs'] = len(train_ds) // params['t_p']['batch_size']
    
    train_results = [torch.zeros((4, params['t_p']['epochs']), device=rank) for _ in range(world_size)]
    test_results = [torch.zeros(4, device=rank) for _ in range(world_size)]
    
    
    params['train_dl'] = DataLoader(
                            train_ds, batch_size=params['t_p']['batch_size'],
                            sampler=DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=100001),
                            shuffle=False, pin_memory=True, persistent_workers=True,
                            num_workers=num_workers, collate_fn=detection_collate(params['t_p']['batch_size']) if task == 'detection' else None
                        )
    
    params['test_dl'] = DataLoader(
                            params['ct_p']['Dataset'].test_ds, batch_size=params['t_p']['batch_size'],
                            sampler=DistributedSampler(params['ct_p']['Dataset'].test_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=100001),
                            shuffle=False, pin_memory=True, persistent_workers=True,
                            num_workers=num_workers, collate_fn=detection_collate(params['t_p']['batch_size']) if task == 'detection' else None
                        )
       
    
    
    train_test = Cls_TrainWorker(rank, params, world_size) if task == 'clf' else Det_TrainWorker(rank, params, world_size)
    
    if rank == 0:
        dist.gather(train_test.train(), train_results)
        dist.gather(train_test.test(), test_results)
    else:
        dist.gather(train_test.train())
        dist.gather(train_test.test())     
    
    dist.barrier()    
    
    
    # shutdown the worker
    del params['train_dl']._iterator
    del params['test_dl']._iterator
    
    gc.collect()
    
    
    if rank == 0:
        train_results = (torch.sum(torch.stack(train_results), dim=0) / world_size).cpu().tolist()
        test_results = (torch.sum(torch.stack(test_results), dim=0) / world_size).cpu().tolist()
        
        conn.send((train_results, test_results))       

    dist.barrier()

    clear_process()

    
    
def train(task: str, params: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    
    train_ds = Subset(params['ct_p']['Dataset'].transformed_trainset, params['ct_p']['Dataset'].labeled_indices)
    
    
    if task == 'det': params['epochs'] = len(train_ds) // params['t_p']['batch_size']
    
    params['train_dl'] = DataLoader(train_ds, batch_size=params['t_p']['batch_size'], shuffle=True, pin_memory=True, collate_fn=detection_collate(params['t_p']['batch_size']) if task == 'detection' else None)
    params['test_dl'] = DataLoader(params['ct_p']['Dataset'].test_ds, batch_size=params['t_p']['batch_size'], shuffle=False, pin_memory=True, collate_fn=detection_collate(params['t_p']['batch_size']) if task == 'detection' else None)
    
    train_test = Cls_TrainWorker(params['ct_p']['device'], params) if task == 'clf' else Det_TrainWorker(params['ct_p']['device'], params)
        
        
    train_results = train_test.train().cpu().tolist()
    test_results = train_test.test().cpu().tolist()
    
    return train_results, test_results