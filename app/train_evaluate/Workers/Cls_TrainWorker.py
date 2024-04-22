
import torch

from models.BBone_Module import Master_Model
from models.modules.LossNet import LossPredLoss
from utils import accuracy_score

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


from typing import Tuple, Dict, Any

import logging
logger = logging.getLogger(__name__)


class Cls_TrainWorker():
    def __init__(self, gpu_id: int, params: Dict[str, Any], world_size: int = 1) -> None:
        
        self.device = torch.device(gpu_id)
        self.iter: int = params['iter']
        
        self.world_size: int = world_size
        self.wandb_run = params['wandb_p'] if 'wandb_p' in params else None

        self.model: Master_Model | DDP = params['ct_p']['Master_Model']
        
        self.dataset_name: str = params['ct_p']['dataset_name']
        self.strategy_name: str = params['strategy_name']
        
        self.epochs = params['t_p']['epochs']
        self.ds_t_p = params['t_p'][self.dataset_name]
        
        self.train_dl: DataLoader = params['train_dl']
        self.test_dl: DataLoader = params['test_dl']
        
        self.backbone_loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.ll_loss_fn = LossPredLoss(self.device).to(self.device)
        self.weight = 1.

        self.score_fn = accuracy_score
        self.best_check_filename = f'app/checkpoints/{self.dataset_name}'        
        self.init_check_filename = f'{self.best_check_filename}/{self.model.module.name if self.world_size > 1 else self.model.name}_init.pth.tar'
        self.check_best_path = f'{self.best_check_filename}/best_{self.strategy_name}_{self.device}.pth.tar'
        
        # RETRAIN FROM SCRATCH THE NETWORK (different from what LL4AL have done)
        self.__load_checkpoint(self.init_check_filename)
        logger.info(' => Loading Initial Checkpoint')


    def init_opt_sched(self):
        self.optimizer = self.ds_t_p['optimizer'](self.model.parameters(), **self.ds_t_p['optim_p'])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[160], gamma=0.1)
    
    
    def __save_checkpoint(self, filename: str) -> None:
        checkpoint = dict(state_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict())
        torch.save(checkpoint, filename)
    
    
    def __load_checkpoint(self, filename: str) -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['state_dict']) if self.world_size > 1 else self.model.load_state_dict(checkpoint['state_dict'])
        self.init_opt_sched()



    def compute_losses(self, module_out: torch.Tensor | None, outputs: torch.Tensor, \
                       labels: torch.Tensor, tot_loss_ce: float, tot_pred_loss: float) -> Tuple[torch.Tensor, float, float]:
                
        loss_ce = self.backbone_loss_fn(outputs, labels)
        backbone = torch.mean(loss_ce)
        
        if module_out == None:
            tot_loss_ce += backbone.item()
            return backbone, tot_loss_ce, tot_pred_loss
        
        elif len(module_out) == 2:
            quantity_loss, mask = module_out
            
            quantity_loss = torch.mean(quantity_loss)
            labeled_loss = torch.mean(loss_ce[mask])
            loss = labeled_loss + quantity_loss
            
            tot_loss_ce += labeled_loss.item()
            tot_pred_loss += quantity_loss.item()
            
            return loss, tot_loss_ce, tot_pred_loss
        
        else:
            loss_weird = self.ll_loss_fn(module_out, loss_ce)
            loss = backbone + self.weight * loss_weird
            # with so I should see the module loss remain stable after the epoch 120

            tot_loss_ce += backbone.item()
            tot_pred_loss += loss_weird.item()
                
            return loss, tot_loss_ce, tot_pred_loss
   
    
    def return_moved_imgs_labs(self, images, labels):
        if self.world_size > 1:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
        else:
            images, labels = images.to(self.device), labels.to(self.device)
        return images, labels
    
    
    def train(self) -> torch.Tensor:
                
        results = torch.zeros((4, self.epochs), device=self.device)
        
        self.model.train()
        
        for epoch in range(self.epochs):
                        
            train_loss, train_loss_ce, train_loss_pred, train_accuracy = .0, .0, .0, .0
            
            if self.world_size > 1: self.train_dl.sampler.set_epoch(epoch) # type: ignore            
                        
            for _, images, labels in self.train_dl:            
                                
                images, labels = self.return_moved_imgs_labs(images, labels)
                                    
                self.optimizer.zero_grad()
                
                outputs, _, module_out = self.model(images, labels=labels, epoch=epoch)
                
                loss, train_loss_ce, train_loss_pred = self.compute_losses(
                        module_out=module_out, outputs=outputs, labels=labels,
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
                
            #logger.info(f' train_loss_ce -> {train_loss_ce}\ttrain_loss_pred -> {train_loss_pred}')
                
            if self.wandb_run != None:
                self.wandb_run.log({
                    'train_accuracy': train_accuracy,
                    'train_loss': train_loss,
                    'train_loss_ce': train_loss_ce,
                    'train_loss_pred': train_loss_pred
                })        
        

            # MultiStepLR
            self.lr_scheduler.step()
            
            # Save checkpoint
            self.__save_checkpoint(self.check_best_path)

        
        return results



    def test(self) -> torch.Tensor:
        test_accuracy, test_loss, test_loss_ce, test_pred_loss = .0, .0, .0, .0
        
        self.model.eval()    

        with torch.inference_mode():
            for _, images, labels in self.test_dl:
                
                images, labels = self.return_moved_imgs_labs(images, labels)
                    
                outputs, _, module_out = self.model(images, labels)

                loss, test_loss_ce, test_pred_loss = self.compute_losses(
                        module_out=module_out, outputs=outputs, labels=labels, 
                        tot_loss_ce=test_loss_ce, tot_pred_loss=test_pred_loss
                    )
                
                test_accuracy += self.score_fn(outputs, labels)
                test_loss += loss.item()

            test_accuracy /= len(self.test_dl)
            test_loss /= len(self.test_dl)
            test_loss_ce /= len(self.test_dl)
            test_pred_loss /= len(self.test_dl)
            
        return torch.tensor((test_accuracy, test_loss, test_loss_ce, test_pred_loss), device=self.device)

        
        