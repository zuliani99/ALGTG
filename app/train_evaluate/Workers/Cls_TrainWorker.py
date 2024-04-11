
import torch

from utils import accuracy_score
from models.ResNet18 import ResNet_LL
from models.Lossnet import LossPredLoss

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


from typing import Tuple, Dict, Any

import logging
logger = logging.getLogger(__name__)


class Cls_TrainWorker():
    def __init__(self, gpu_id: int, params: Dict[str, Any], world_size: int = 1) -> None:
        
        self.device = torch.device(gpu_id)
        self.iter: int = params['iter']
        
        self.LL: bool = params['LL']
        self.world_size: int = world_size
        self.wandb_run = params['wandb_p'] if 'wandb_p' in params else None

        self.model: ResNet_LL | DDP = params['ct_p']['Model_train']
        
        self.dataset_name: str = params['ct_p']['dataset_name']
        self.method_name: str = params['method_name']
        
        self.epochs = params['t_p']['epochs']
        self.ds_t_p = params['t_p']['ds_params'][self.dataset_name]
        
        self.train_dl: DataLoader = params['train_dl']
        self.test_dl: DataLoader = params['test_dl']
        
        self.loss_fn = dict(
            backbone = torch.nn.CrossEntropyLoss(reduction='none').to(self.device),
            module = LossPredLoss(self.device).to(self.device)
        )
                
        self.score_fn = accuracy_score
        
        self.best_check_filename = f'app/checkpoints/{self.dataset_name}'        
        self.init_check_filename = f'{self.best_check_filename}_init.pth.tar'
        logger.info(' => Loading Initial Checkpoint')
        self.__load_checkpoint(self.init_check_filename)
        logger.info(' DONE')


    def init_opt_sched(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.ds_t_p['lr'], momentum=self.ds_t_p['mom'], weight_decay=self.ds_t_p['w_d'])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[160], gamma=0.1)
    
    
    def __save_checkpoint(self, filename: str) -> None:
        checkpoint = dict(state_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict())
        torch.save(checkpoint, filename)
    
    
    def __load_checkpoint(self, filename: str) -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['state_dict']) if self.world_size > 1 else self.model.load_state_dict(checkpoint['state_dict'])
        self.init_opt_sched()



    def compute_losses(self, weight: float, pred_loss: torch.Tensor, outputs: torch.Tensor, \
                       labels: torch.Tensor, tot_loss_ce: float, tot_pred_loss: float) -> Tuple[torch.Tensor, float, float]:
                
        loss_ce = self.loss_fn['backbone'](outputs, labels)
        backbone = torch.mean(loss_ce)
        if self.LL and weight:
            loss_weird = self.loss_fn['module'](pred_loss, loss_ce)
            loss = backbone + loss_weird

            tot_loss_ce += backbone.item()
            tot_pred_loss += loss_weird.item()
            
            return loss, tot_loss_ce, tot_pred_loss
        else:            
            tot_loss_ce += backbone.item()
            return backbone, tot_loss_ce, tot_pred_loss    
        
    
    
    
    def train(self) -> torch.Tensor:
                
        weight = 1.
        check_best_path = f'{self.best_check_filename}/best_{self.method_name}_{self.device}.pth.tar'
        results = torch.zeros((4, self.epochs), device=self.device)
        
        if self.iter > 1: self.__load_checkpoint(check_best_path)
            
        self.model.train()
        
        for epoch in range(self.epochs):
            
            train_loss, train_loss_ce, train_loss_pred, train_accuracy = .0, .0, .0, .0
            
            if self.LL and epoch == 121: weight = 0
                        
            if self.world_size > 1: self.train_dl.sampler.set_epoch(epoch) # type: ignore            
                        
            for _, images, labels in self.train_dl:                
                
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                    
                self.optimizer.zero_grad()
                
                outputs, _, pred_loss = self.model(images)
                
                loss, train_loss_ce, train_loss_pred  = self.compute_losses(
                        weight=weight, pred_loss=pred_loss, outputs=outputs, labels=labels,
                        tot_loss_ce=train_loss_ce, tot_pred_loss=train_loss_pred
                    )
                                
                loss.backward()

                self.optimizer.step()
                
                train_loss += loss.item()
                train_accuracy += self.score_fn(outputs, labels)


            train_accuracy /= len(self.train_dl)
            train_loss /= len(self.train_dl)
            train_loss_ce /= len(self.train_dl)
            train_loss_pred /= len(self.train_dl)
                        
            
            for pos, metric in zip(range(results.shape[0]), [train_accuracy, train_loss, train_loss_ce, train_loss_pred]):
                results[pos][epoch] = metric
                
            if self.wandb_run != None:
                self.wandb_run.log({
                    'train_accuracy': train_accuracy,
                    'train_loss': train_loss,
                    'train_loss_ce': train_loss_ce,
                    'train_loss_pred': train_loss_pred
                })        
        

            #MultiStepLR
            self.lr_scheduler.step()
            
            self.__save_checkpoint(check_best_path)
                
        # load best checkpoint
        self.__load_checkpoint(check_best_path)
        #self.check_best_path = check_best_path
        
        return results



    def test(self) -> torch.Tensor:
        test_accuracy, test_loss, test_loss_ce, test_pred_loss = .0, .0, .0, .0
        
        self.model.eval()        

        with torch.inference_mode():
            for _, images, labels in self.test_dl:
                
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                    
                outputs, _, pred_loss = self.model(images)

                loss, test_loss_ce, test_pred_loss = self.compute_losses(
                        weight=1, pred_loss=pred_loss, outputs=outputs, labels=labels, 
                        tot_loss_ce=test_loss_ce, tot_pred_loss=test_pred_loss
                    )
                
                test_accuracy += self.score_fn(outputs, labels)
                test_loss += loss.item()

            test_accuracy /= len(self.test_dl)
            test_loss /= len(self.test_dl)
            test_loss_ce /= len(self.test_dl)
            test_pred_loss /= len(self.test_dl)
            
        return torch.tensor((test_accuracy, test_loss, test_loss_ce, test_pred_loss), device=self.device)

        
        