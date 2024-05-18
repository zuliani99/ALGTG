
import torch
import torch.nn as nn

from models.BBone_Module import Master_Model
from models.modules.LossNet import LossPredLoss
from utils import accuracy_score, log_assert

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from typing import Tuple, Dict, Any, List

import logging
logger = logging.getLogger(__name__)


class Cls_TrainWorker():
    def __init__(self, gpu_id: int, params: Dict[str, Any], world_size: int = 1) -> None:
        
        self.device = torch.device(gpu_id)
        self.iter: int = params["iter"]
        
        self.world_size: int = world_size

        self.model: Master_Model | DDP = params["ct_p"]["Master_Model"]
        
        self.dataset_name: str = params["ct_p"]["dataset_name"]
        self.strategy_name: str = params["strategy_name"]
        self.method_name: str = self.strategy_name.split(f'{self.model.name}_')[1]
        
        self.epochs = params["t_p"]["epochs"]
        self.ds_t_p = params["t_p"][self.dataset_name]
        
        self.train_dl: DataLoader = params["train_dl"]
        self.test_dl: DataLoader = params["test_dl"]
        
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.mse_loss_fn = nn.MSELoss(reduction='none').to(self.device)
        self.l1_loss_fn = nn.L1Loss(reduction='none').to(self.device)
        self.kld_loss_fn = nn.KLDivLoss(reduction='batchmean').to(self.device)
        self.ll_loss_fn = LossPredLoss(self.device).to(self.device)
        
        self.score_fn = accuracy_score
        self.init_check_filename = f'app/checkpoints/{self.dataset_name}/{self.model.module.name if self.world_size > 1 else self.model.name}_init.pth.tar'
        self.check_best_path = f'app/checkpoints/{self.dataset_name}/best_{self.strategy_name}_{self.device}.pth.tar'
        
        # RETRAIN FROM SCRATCH THE NETWORK (different from what LL4AL have done)
        self.__load_checkpoint(self.init_check_filename)
        self.init_opt_sched()
        self.i = 0


    def init_opt_sched(self):
        optimizers = self.ds_t_p["optimizers"]
        module_name = self.model.added_module_name if self.model.added_module == None else self.model.added_module_name.split('_')[0]
        self.optimizers: List[torch.optim.SGD | torch.optim.Adam] = []
        self.optimizers.append(optimizers["backbone"]["type"][module_name](self.model.backbone.parameters(), **optimizers["backbone"]["optim_p"][module_name]))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizers[0], milestones=[160], gamma=0.1)
        if module_name != None:
            self.optimizers.append(optimizers["modules"]["type"][module_name](self.model.added_module.parameters(), **optimizers["modules"]["optim_p"][module_name]))
            
    
    def __save_checkpoint(self, filename: str) -> None:
        logger.info(f' => Saving {filename} Checkpoint')
        checkpoint = dict(state_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict())
        torch.save(checkpoint, filename)
    
    
    def __load_checkpoint(self, filename: str) -> None:
        logger.info(f' => Loading {filename} Checkpoint')
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.module.load_state_dict(checkpoint["state_dict"]) if self.world_size > 1 else self.model.load_state_dict(checkpoint["state_dict"])
        

    def compute_losses(self, weight: float, module_out: torch.Tensor | None, outputs: torch.Tensor, \
                       labels: torch.Tensor, tot_loss_ce: float, tot_pred_loss: float, epoch: int,
                       tidal: None | Tuple[torch.Tensor, torch.Tensor , int] = None,) -> Tuple[torch.Tensor, float, float]:
                
        ce_loss = self.ce_loss_fn(outputs, labels)
        backbone = torch.mean(ce_loss)
        self.i += 1
            
        if module_out == None:
            tot_loss_ce += backbone.item()
            return backbone, tot_loss_ce, tot_pred_loss
        
        elif self.method_name.split('_')[0] == 'GTG' and self.model.added_module_name == 'GTGModule':
                        
            (pred_entr, true_entr), labelled_mask = module_out
            if self.i%20 == 0: logger.info(f'y_pred {pred_entr}\ny_true {true_entr}')
                
            entr_loss = weight * self.mse_loss_fn(pred_entr, true_entr.detach())

            ################################################
            lab_ce_loss = torch.mean(ce_loss[labelled_mask])
            entr_loss = torch.mean(entr_loss)
            ################################################
            
            loss = lab_ce_loss + entr_loss
            
            tot_loss_ce += lab_ce_loss.item()
            tot_pred_loss += entr_loss.item()
            self.i += 1
            
            
            return loss, tot_loss_ce, tot_pred_loss
            
        
        elif self.method_name in ['LearningLoss', 'TAVAAL'] or 'GTG_off' in self.method_name:
            loss_weird = weight * self.ll_loss_fn(module_out, ce_loss)
            loss = backbone + loss_weird

            tot_loss_ce += backbone.item()
            tot_pred_loss += loss_weird.item()
                
            return loss, tot_loss_ce, tot_pred_loss
        
        elif self.method_name == 'TiDAL':
            log_assert(tidal != None, 'TiDAL parameters are None')
            idxs, moving_prob, epoch = tidal # type: ignore
            moving_prob = moving_prob.to(self.device)
            
            moving_prob = (moving_prob * epoch + torch.softmax(outputs, dim=1) * 1) / (epoch + 1)
            self.train_dl.dataset.moving_prob[idxs.tolist(), :] = moving_prob.detach().cpu() # type: ignore
                        
            m_module_loss = weight * self.kld_loss_fn(F.log_softmax(module_out, 1), moving_prob.detach())
            loss = backbone + m_module_loss
            
            tot_loss_ce += backbone.item()
            tot_pred_loss += m_module_loss.item()
                
            return loss, tot_loss_ce, tot_pred_loss

        else: raise AttributeError('Invalid method_name')
    
    
    def return_moved_imgs_labs(self, images, labels):
        if self.world_size > 1:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
        else:
            images, labels = images.to(self.device), labels.to(self.device)
        return images, labels
    
    
    def train(self) -> torch.Tensor:
        
        weight = 1.
        results = torch.zeros((4, self.epochs), device=self.device)
        
        ###########################################################
        #idxs, images, labels, moving_prob = next(iter(self.train_dl))
        #images, labels = self.return_moved_imgs_labs(images, labels)
        ###########################################################
                
        self.model.train()
                
        for epoch in range(self.epochs):
                        
            train_loss, train_loss_ce, train_loss_pred, train_accuracy = .0, .0, .0, .0
            
            if self.world_size > 1: self.train_dl.sampler.set_epoch(epoch) # type: ignore  
            if epoch >= 120: weight = 0.
             
            for idxs, images, labels, moving_prob in self.train_dl:
                images, labels = self.return_moved_imgs_labs(images, labels)
                
                for optimizer in self.optimizers: optimizer.zero_grad(set_to_none=True)
                    
                outputs, _, module_out = self.model(images, epoch=epoch, labels=labels)
                                                                        
                loss, train_loss_ce, train_loss_pred = self.compute_losses(
                            weight=weight, module_out=module_out, outputs=outputs, labels=labels,
                            tot_loss_ce=train_loss_ce, tot_pred_loss=train_loss_pred, epoch=epoch, 
                            tidal=(idxs, moving_prob, epoch),
                        )  
                    
                loss.backward()                
                for optimizer in self.optimizers: optimizer.step()

                train_loss += loss.item()
                train_accuracy += self.score_fn(outputs, labels)
                

            train_accuracy /= len(self.train_dl)
            train_loss /= len(self.train_dl)
            train_loss_ce /= len(self.train_dl)
            train_loss_pred /= len(self.train_dl)
            
            logger.info(f' Epoch: {epoch} | train_accuracy -> {train_accuracy}\ttrain_loss -> {train_loss}\ttrain_loss_ce -> {train_loss_ce}\ttrain_pred -> {train_loss_pred}')
            
            for pos, metric in zip(range(results.shape[0]), [train_accuracy, train_loss, train_loss_ce, train_loss_pred]):
                results[pos][epoch] = metric

            # MultiStepLR
            self.lr_scheduler.step()
            
        # Save checkpoint
        self.__save_checkpoint(self.check_best_path)

        
        return results



    def test(self) -> torch.Tensor:
        
        #self.__load_checkpoint(self.check_best_path)
        test_accuracy = .0
        
        self.model.eval()    

        with torch.inference_mode():
            for _, images, labels in self.test_dl:
                
                images, labels = self.return_moved_imgs_labs(images, labels)
                outputs, _  = self.model.backbone(images)
                test_accuracy += self.score_fn(outputs, labels)

            test_accuracy /= len(self.test_dl)
            
        return torch.tensor([test_accuracy], device=self.device)
