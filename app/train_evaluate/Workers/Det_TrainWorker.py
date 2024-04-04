
'''
https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-3-distributed-data-parallel-d26e93f17c62
https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
https://pytorch.org/docs/stable/multiprocessing.html
https://discuss.pytorch.org/t/how-can-i-get-returns-from-a-function-in-distributed-data-parallel/120067/2
https://tuni-itc.github.io/wiki/Technical-Notes/Distributed_dataparallel_pytorch/#distributeddataparallel-as-a-batch-job-in-the-servers
https://medium.com/@ramyamounir/distributed-data-parallel-with-slurm-submitit-pytorch-168c1004b2ca
'''

import pickle
import torch

from ...models.Lossnet import LossPredLoss
from ...models.ssd_pytorch.SSD import SSD
from ...models.ssd_pytorch.detect_eval import do_python_eval, write_voc_results_file
from ...models.ssd_pytorch.ssd_layers.modules.multibox_loss import MultiBoxLoss

from torch.utils.data import DataLoader

from typing import Dict, Any
import os
import numpy as np

from ...utils import cycle, get_output_dir


class Det_TrainWorker():
    def __init__(self, gpu_id: int, params: Dict[str, Any], world_size: int = 1) -> None:
        
        self.device = torch.device(gpu_id)
        self.iter: int = params['iter']
        self.params = params
        self.epochs = self.params['t_p']['epochs']
        
        
        self.LL: bool = params['LL']
        self.world_size: int = world_size
        self.model: SSD = params['ct_p']['model']
        
        self.dataset_name: str = params['ct_p']['dataset_name']
        self.method_name: str = params['method_name']
        
        self.train_dl: DataLoader = params['train_dl']
        self.test_dl: DataLoader = params['test_dl']
        
        
        self.loss_fn = MultiBoxLoss(self.params['ct_p']['Dataset'].n_classes, 0.5, True, 0, True, 3, 0.5, False, self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[240], gamma=0.1)
        self.loss_weird = LossPredLoss(self.device).to(self.device)
        
        self.best_check_filename = f'app/checkpoints/{self.dataset_name}'
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

        # loss counters
        loc_loss, conf_loss = 0, 0
        train_target_loss, train_loss_weird = 0, 0
        
        epoch = 0
        step_index = 0
                
        check_best_path = f'{self.best_check_filename}/best_{self.method_name}_{self.device}.pth.tar'
        results = torch.zeros((4, self.epochs), device=self.device)
        
        if self.iter > 1: self.__load_checkpoint(check_best_path)
        
        if self.world_size > 1: self.train_dl.sampler.set_epoch(epoch) # type: ignore
        batch_iterator = iter(cycle(self.train_dl))

        for iteration in range(self.params['t_p']['start_iter'], self.params['t_p']['max_iter']):
            if iteration != 0 and (iteration % self.epochs == 0):
                # reset epoch loss counters
                
                self.__save_checkpoint(check_best_path)
                self.scheduler.step()
                
                for pos, metric in zip(range(results.shape[0]), [train_target_loss / iteration, loc_loss, conf_loss, train_loss_weird / iteration]): 
                    results[pos][epoch] = metric
                
                loc_loss, conf_loss = 0, 0
                epoch += 1

            if iteration in self.params['t_p']['lr_steps']:
                step_index += 1
                self.adjust_learning_rate(step_index)

            # load train data
            images, targets = next(batch_iterator)
            images, targets = images.to(self.device, non_blocking=True), [ann.to(self.device, non_blocking=True) for ann in targets]


            self.optimizer.zero_grad()
            
            # forward
            outputs, out_weird, _ = self.model(images)
            
            loss_l, loss_c = self.loss_fn(outputs, targets)
            target_loss = loss_l + loss_c

            module_loss = self.loss_weird(out_weird, target_loss.reshape(1))
            loss = target_loss + module_loss
            

            loss.backward()
            self.optimizer.step()

            train_target_loss += loss.item()
            train_loss_weird += module_loss
            loc_loss += loss_l.data
            conf_loss += loss_c.data
            
            torch.cuda.empty_cache()

        self.__load_checkpoint(check_best_path)

        return results



    def test(self):
        
        self.model.change_phase()
        self.model.eval()

        # evaluation
        num_images = len(self.test_dl.dataset)
        # all detections are collected into:

        all_boxes = [[[] for _ in range(num_images)] for _ in range(self.params['ct_p']['Datasets'].n_classes)]

        output_dir = get_output_dir(f'results/{self.params['ct_p']['timestamp']}/{self.params['ct_p']['datasetname']}/{self.params['t_p']['trail']}/{self.method_name}', 'test')        
        det_file = os.path.join(output_dir, 'detections.pkl')
        
        total_loss, total_target_loss, total_loss_weird = .0, .0, .0

        with torch.no_grad(): # Allow inference mode
            for i in range(num_images):
                im, gt, h, w = self.test_dl.dataset.pull_item(i)

                x = im.unsqueeze(0).to(self.device, non_blocking=True)
                outputs, out_weird, _ = self.model(x)
                
                loss_l, loss_c = self.loss_fn(outputs, gt)
                target_loss = loss_l + loss_c

                module_loss = self.loss_weird(out_weird, target_loss.reshape(1))
                loss = target_loss + module_loss
                
                total_loss += loss.item()
                total_target_loss += target_loss
                total_loss_weird += module_loss

                outputs = outputs.data
                # skip j = 0, because it's the background class
                for j in range(1, outputs.size(1)):
                    dets = outputs[0, j, :]
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

        total_loss /= num_images
        total_target_loss /= num_images
        total_loss_weird /= num_images
        
        with open(det_file, 'wb') as f: pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        write_voc_results_file(all_boxes, self.test_dl.dataset, self.params['ct_p']['Dataset'].dataset_path)
        mAP = do_python_eval(self.params['ct_p']['Dataset'].dataset_path, output_dir)

        torch.cuda.empty_cache()

        return torch.tensor((mAP, total_loss, total_target_loss, total_loss_weird), device=self.device)
    


    def adjust_learning_rate(self, step):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        # lr = args.lr * (gamma ** (step))
        lr = 0.01 * (0.1 ** (step))
        for param_group in self.optimizer.param_groups[:-1]:
            param_group['lr'] = lr