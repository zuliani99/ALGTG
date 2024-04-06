
'''
https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-3-distributed-data-parallel-d26e93f17c62
https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
https://pytorch.org/docs/stable/multiprocessing.html
https://discuss.pytorch.org/t/how-can-i-get-returns-from-a-function-in-distributed-data-parallel/120067/2
https://tuni-itc.github.io/wiki/Technical-Notes/Distributed_dataparallel_pytorch/#distributeddataparallel-as-a-batch-job-in-the-servers
https://medium.com/@ramyamounir/distributed-data-parallel-with-slurm-submitit-pytorch-168c1004b2ca
'''

import torch

from utils import accuracy_score
from models.ResNet18 import ResNet
from models.Lossnet import LossPredLoss

from torch.utils.data import DataLoader

from typing import Tuple, Dict, Any


class Cls_TrainWorker():
    def __init__(self, gpu_id: int, params: Dict[str, Any], world_size: int = 1) -> None:
        
        self.device = torch.device(gpu_id)
        self.iter: int = params['iter']
        
        self.LL: bool = params['LL']
        self.world_size: int = world_size
        self.model: ResNet = params['ct_p']['Model']
        self.epochs = params['t_p']['epochs']
        
        self.dataset_name: str = params['ct_p']['dataset_name']
        self.method_name: str = params['method_name']
        
        self.train_dl: DataLoader = params['train_dl']
        self.test_dl: DataLoader = params['test_dl']
        
        
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.score_fn = accuracy_score
        self.loss_weird = LossPredLoss(self.device).to(self.device)
        
        self.best_check_filename = f'app/checkpoints/{self.dataset_name}'
        self.init_check_filename = f'{self.best_check_filename}_init_checkpoint.pth.tar'
        
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[160], gamma=0.1)

    
    
    def __save_checkpoint(self, filename: str) -> None:
        checkpoint = {
            'state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
        }
        torch.save(checkpoint, filename)
    
    
    
    def __load_checkpoint(self, filename: str) -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['state_dict']) if self.world_size > 1 else self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[160], gamma=0.1)



    def compute_losses(self, weight: float, out_weird: torch.Tensor, outputs: torch.Tensor, \
        labels: torch.Tensor, tot_loss_ce: float, tot_loss_weird: float) -> Tuple[torch.Tensor, float, float]:
        
        loss_ce = self.loss_fn(outputs, labels)
        
        if self.LL and weight:    
            loss_weird = self.loss_weird(out_weird, loss_ce)
            loss_ce = torch.mean(loss_ce)
            loss = loss_ce + loss_weird
                    
            tot_loss_ce += loss_ce.cpu().item()
            tot_loss_weird += loss_weird.cpu().item()
        else:
            loss = torch.mean(loss_ce)
            if self.LL: tot_loss_ce += loss.item()
            
        return loss, tot_loss_ce, tot_loss_weird               
        
    
    
    
    def train(self) -> torch.Tensor:
                
        weight = 1.
        check_best_path = f'{self.best_check_filename}/best_{self.method_name}_{self.device}.pth.tar'
        results = torch.zeros((4, self.epochs), device=self.device)
        
        
        if self.iter > 1: self.__load_checkpoint(check_best_path)
                    
    
        for epoch in range(self.epochs):

            train_loss, train_loss_ce, train_loss_weird, train_accuracy = .0, .0, .0, .0
            
            if self.LL and epoch == 121: weight = 0

            
            if self.world_size > 1: self.train_dl.sampler.set_epoch(epoch) # type: ignore
            self.model.train()
            
                        
            for _, images, labels in self.train_dl:                
                                    
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                outputs, _, out_weird = self.model(images)
                
                loss, train_loss_ce, train_loss_weird  = self.compute_losses(
                        weight, out_weird, outputs, labels, train_loss_ce, train_loss_weird
                    )
                                
                loss.backward()

                self.optimizer.step()
                
                train_loss += loss.item()
                train_accuracy += self.score_fn(outputs, labels)


            train_accuracy /= len(self.train_dl)
            train_loss /= len(self.train_dl)
            train_loss_ce /= len(self.train_dl)
            train_loss_weird /= len(self.train_dl)
            

            
            #for pos, metric in zip(range(4), [train_loss, train_loss_ce, train_loss_weird, train_accuracy]):
            for pos, metric in zip(range(results.shape[0]), [train_loss, train_loss_ce, train_loss_weird, train_accuracy]):
                results[pos][epoch] = metric

            #MultiStepLR
            self.scheduler.step()
            
            self.__save_checkpoint(check_best_path)
                
        
        
        # load best checkpoint
        self.__load_checkpoint(check_best_path)
        
        return results



    def test(self) -> torch.Tensor:
        tot_loss, tot_loss_ce, tot_loss_weird, tot_accuracy = .0, .0, .0, .0
        
        self.model.eval()

        with torch.no_grad(): # Allow inference mode
            for _, images, labels in self.test_dl:
                
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs, _, out_weird  = self.model(images)

                loss, tot_loss_ce, tot_loss_weird = self.compute_losses(
                        1, out_weird, outputs, labels, tot_loss_ce, tot_loss_weird
                    )
                
                tot_accuracy += self.score_fn(outputs, labels)
                tot_loss += loss.item()


            tot_accuracy /= len(self.test_dl)
            tot_loss /= len(self.test_dl)
            tot_loss_ce /= len(self.test_dl)
            tot_loss_weird /= len(self.test_dl)
            
        return torch.tensor((tot_accuracy, tot_loss, tot_loss_ce, tot_loss_weird), device=self.device)

        
        