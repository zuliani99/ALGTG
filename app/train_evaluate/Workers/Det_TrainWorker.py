

import pickle
import torch

from models.Lossnet import LossPredLoss
from models.ssd_pytorch.SSD import SSD
from models.ssd_pytorch.detect_eval import do_python_eval, write_voc_results_file
from models.ssd_pytorch.ssd_layers.modules.multibox_loss import MultiBoxLoss
from datasets_creation.Detection import Det_Datasets

from torch.utils.data import DataLoader

from typing import Dict, Any, List
import os
import numpy as np

from utils import create_directory, cycle

import logging
logger = logging.getLogger(__name__)


class Det_TrainWorker():
    def __init__(self, gpu_id: int, params: Dict[str, Any], world_size: int = 1) -> None:
        
        self.device = torch.device(gpu_id)
        self.iter: int = params['iter']
        self.ct_p, self.t_p = params['ct_p'], params['t_p']
        self.epochs: int = self.t_p ['epochs']
        self.dataset: Det_Datasets = self.ct_p['Dataset']
        
        self.world_size: int = world_size
        self.model: SSD = self.ct_p['Model']
        
        self.method_name: str = params['method_name']
        
        self.train_dl: DataLoader = params['train_dl']
        self.test_dl: DataLoader = params['test_dl']
        
        
        self.loss_fn = MultiBoxLoss(self.dataset.n_classes, 0.5, True, 0, True, 3, 0.5, False, self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[240], gamma=0.1)
        self.loss_weird = LossPredLoss(self.device).to(self.device)
        
        self.best_check_filename = f'app/checkpoints/{self.ct_p['dataset_name']}'
        self.init_check_filename = f'{self.best_check_filename}_init_checkpoint.pth.tar'
    
    
    
    def __save_checkpoint(self, filename: str) -> None:
        checkpoint = { 'state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict() }
        torch.save(checkpoint, filename)
    
    
    
    def __load_checkpoint(self, filename: str) -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['state_dict']) if self.world_size > 1 else self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[240], gamma=0.1)

    

    def train(self):
        self.model.train()
        
        #LL = False

        # loss counters
        loc_loss, conf_loss, train_loss, train_loss_weird = .0, .0, .0, .0
        
        epoch = 0
        step_index = 0
                
        check_best_path = f'{self.best_check_filename}/best_{self.method_name}_{self.device}.pth.tar'
        #results = torch.zeros((4 if LL else 3, self.epochs), device=self.device)
        results = torch.zeros((4, self.epochs), device=self.device)
        
        if self.iter > 1: self.__load_checkpoint(check_best_path)
        
        if self.world_size > 1: self.train_dl.sampler.set_epoch(epoch) # type: ignore
        batch_iterator = iter(cycle(self.train_dl))

        for iteration in range(0, self.t_p['max_iter']): #120000
            
            losses = torch.empty((0, 2))
            
            if iteration != 0 and (iteration % self.epochs == 0):
                # reset epoch loss counters
                
                self.__save_checkpoint(check_best_path)
                self.scheduler.step()
                
                logger.info(f' Epoch {epoch} | Iteration {iteration} -> train_loss: {train_loss / self.epochs}\tloc_loss: {loc_loss / self.epochs}\tconf_loss: {conf_loss / self.epochs}\ttrain_loss_weird: {train_loss_weird / self.epochs}')
                
                for pos, metric in zip(range(results.shape[0]), [train_loss / self.epochs,
                                                                 loc_loss / self.epochs, 
                                                                 conf_loss / self.epochs, 
                                                                 train_loss_weird / self.epochs]): 
                    if metric != 0.0: results[pos][epoch] = metric
                
                loc_loss, conf_loss = .0, .0
                train_loss, train_loss_weird = .0, .0
                epoch += 1

            if iteration in self.t_p['lr_steps']:
                step_index += 1
                self.adjust_learning_rate(step_index)

            # load train data
            _, images, targets = next(batch_iterator)
            
            images = torch.autograd.Variable(images.to(self.device, non_blocking=True))
            targets = [torch.autograd.Variable(ann.to(self.device, non_blocking=True), requires_grad=False) for ann in targets]

            
            self.optimizer.zero_grad()
            
            
            # forward
            outputs, out_weird, _ = self.model(images)
            loss_l, loss_c = self.loss_fn(outputs, targets)
            initial_loss = loss_l + loss_c
            
            
            # FIX THE LERNING THAT IS NOT GOING DOWN
            # WANDB COULD BE USEFUL NOW
            '''
            loc_data, conf_data, priors = outputs
            for idx_image in range(images.shape[0]):
                output_tuple = (
                    loc_data[idx_image,:,:].unsqueeze(0),
                    conf_data[idx_image,:,:].unsqueeze(0),
                    priors
                )
                losses = torch.cat((losses, torch.tensor([self.loss_fn(output_tuple, [targets[idx_image]])]))) # [32,2]
            target_loss = torch.sum(losses, dim = 1) # -> [32,1]
            module_loss = self.loss_weird(out_weird, target_loss)
            tuned_loss = torch.mean(target_loss) + module_loss

            logger.info(f'original loss: {initial_loss} - loss_weired fianl: {tuned_loss}')'''


            initial_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., norm_type=2)
            self.optimizer.step()

            train_loss += initial_loss.item()
            #if LL: train_loss_weird += module_loss.item()
            #train_loss_weird += module_loss.item()
            loc_loss += loss_l#torch.mean(losses[:, 0]).item()
            conf_loss += loss_c#torch.mean(losses[:, 1]).item()
            
            torch.cuda.empty_cache()

        self.__load_checkpoint(check_best_path)

        return results



    def test(self):
        
        self.model.change_phase()
        self.model.eval()

        # evaluation
        num_images = len(self.test_dl.dataset)
        # all detections are collected into:

        all_boxes: List[List[np.ndarray | None]] = [[ None for _ in range(num_images)] for _ in range(self.dataset.n_classes)]

        output_dir = f'results/{self.ct_p['timestamp']}/{self.ct_p['dataset_name']}/{self.ct_p['trial']}/{self.method_name}/test'
        create_directory(output_dir)        
        det_file = os.path.join(output_dir, 'detections.pkl')
        
        with torch.no_grad(): # Allow inference mode
            logger.info(' => Detection Phase On Going...')
            for i in range(num_images):
                im, _, h, w = self.test_dl.dataset.pull_item(i)
                x = im.unsqueeze(0).to(self.device, non_blocking=True)
                detection, _, _ = self.model(x)
                
                # skip j = 0, because it's the background class
                for j in range(1, detection.size(1)):
                    dets = detection[0, j, :]
                    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                    dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.size(0) == 0: continue
                    boxes = dets[:, 1:]
                    boxes[:, 0] *= w
                    boxes[:, 2] *= w
                    boxes[:, 1] *= h
                    boxes[:, 3] *= h
                    scores = dets[:, 0].cpu().numpy()
                    cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
                    all_boxes[j][i] = cls_dets

                #print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))
                
        logger.info(' => Saving Detections')
        with open(det_file, 'wb') as f: pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        logger.info(' DONE\n')

        logger.info(' => Evaluating detections...')
        write_voc_results_file(all_boxes, self.test_dl.dataset, self.dataset.dataset_path, self.dataset.classes)
        mAP = do_python_eval(self.dataset.classes, self.dataset.dataset_path, output_dir)
        logger.info(' DONE\n')
        
        torch.cuda.empty_cache()

        return torch.tensor((mAP), device=self.device)
    


    def adjust_learning_rate(self, step):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        # lr = args.lr * (gamma ** (step))
        lr = 0.01 * (0.1 ** (step))
        for param_group in self.optimizer.param_groups:#[:-1]: # discard the LL module????
            param_group['lr'] = lr