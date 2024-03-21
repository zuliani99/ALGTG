
'''
https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-3-distributed-data-parallel-d26e93f17c62
https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
https://pytorch.org/docs/stable/multiprocessing.html
https://discuss.pytorch.org/t/how-can-i-get-returns-from-a-function-in-distributed-data-parallel/120067/2
https://tuni-itc.github.io/wiki/Technical-Notes/Distributed_dataparallel_pytorch/#distributeddataparallel-as-a-batch-job-in-the-servers
https://medium.com/@ramyamounir/distributed-data-parallel-with-slurm-submitit-pytorch-168c1004b2ca
'''

import torch

from utils import init_weights_apply
from ResNet18 import LearningLoss, ResNet_Weird
from torch.utils.data import DataLoader

from typing import Tuple, Dict, Any


class TrainWorker():
    def __init__(self, gpu_id: str, params: Dict[str, Any], world_size: int = 1) -> None:
        
        self.device = torch.device(gpu_id)
        
        self.LL: bool = params['LL']
        self.world_size = world_size
        self.patience: int = params['patience'],
        self.model: ResNet_Weird = params['model']
        
        self.dataset_name: str = params['dataset_name']
        self.method_name: str = params['method_name']
        
        self.train_dl: DataLoader = params['train_dl']
        self.val_dl: DataLoader = params['val_dl']
        self.test_dl: DataLoader = params['test_dl']
        
        
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.score_fn: function = params['score_fn']
        self.loss_weird = LearningLoss(self.device).to(self.device)
        
        self.best_check_filename = f'app/checkpoints/{self.dataset_name}'
        self.init_check_filename = f'{self.best_check_filename}_init_checkpoint.pth.tar'
        

        self.model.apply(init_weights_apply)
            
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[160], gamma=0.1)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    
    
    def __save_checkpoint(self, filename: str) -> None:
        checkpoint = {
            'state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, filename)
    
    
    
    def __load_checkpoint(self, filename: str) -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['state_dict']) if self.world_size > 1 else self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])



    def compute_losses(self, weight: int, out_weird: torch.Tensor, outputs: torch.Tensor, \
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




    def evaluate(self, dataloader: DataLoader, weight: int) -> Tuple[float, float, float, float]:
                
        tot_loss, tot_loss_ce, tot_loss_weird, tot_accuracy = .0, .0, .0, .0
        
        self.model.eval()

        with torch.no_grad(): # Allow inference mode
            for _, images, labels in dataloader:
                
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs, _, out_weird, _ = self.model(images)

                loss, tot_loss_ce, tot_loss_weird = self.compute_losses(
                        weight, out_weird, outputs, labels, tot_loss_ce, tot_loss_weird
                    )
                
                tot_accuracy += self.score_fn(outputs, labels)
                tot_loss += loss.item()


            tot_accuracy /= len(dataloader)
            tot_loss /= len(dataloader)
            tot_loss_ce /= len(dataloader)
            tot_loss_weird /= len(dataloader)
            
            
        return tot_accuracy, tot_loss, tot_loss_ce, tot_loss_weird
    
    
    
    def train_evaluate(self, epochs: int) -> torch.Tensor:
                
        weight = 1.
        check_best_path = f'{self.best_check_filename}/best_{self.method_name}_{self.device}.pth.tar'
        best_val_accuracy = 0.0
        results = torch.zeros((8, epochs), device=self.device)
    
        for epoch in range(epochs):

            train_loss, train_loss_ce, train_loss_weird, train_accuracy = .0, .0, .0, .0
            
            if self.LL and epoch == 121: weight = 0

            
            if self.world_size > 1:
                self.train_dl.sampler.set_epoch(epoch)
            self.model.train()
            
                        
            for _, images, labels in self.train_dl:                
                                    
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                outputs, _, out_weird, _ = self.model(images)
                
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
            

            
            for pos, metric in zip(range(4), [train_loss, train_loss_ce, train_loss_weird, train_accuracy]):
                results[pos][epoch] = metric
            
            # evaluating using the validation set
            val_accuracy, val_loss, val_loss_ce, val_loss_weird = self.evaluate(self.val_dl, weight)
            
            #MultiStepLR
            self.scheduler.step()
            
            for pos, metric in zip(range(4,8), [val_loss, val_loss_ce, val_loss_weird, val_accuracy]):
                results[pos][epoch] = metric
                
            
            if(val_accuracy > best_val_accuracy):
                best_val_accuracy = val_accuracy
                
                self.__save_checkpoint(check_best_path)
                
        
        
        # load best checkpoint
        self.__load_checkpoint(check_best_path)
        
        return results



    def test(self) -> torch.Tensor:           
        return torch.tensor(self.evaluate(self.test_dl, weight=1), device=self.device)

        
        