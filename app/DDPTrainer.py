
'''
https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-3-distributed-data-parallel-d26e93f17c62
https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
https://pytorch.org/docs/stable/multiprocessing.html
https://discuss.pytorch.org/t/how-can-i-get-returns-from-a-function-in-distributed-data-parallel/120067/2
https://tuni-itc.github.io/wiki/Technical-Notes/Distributed_dataparallel_pytorch/#distributeddataparallel-as-a-batch-job-in-the-servers
https://medium.com/@ramyamounir/distributed-data-parallel-with-slurm-submitit-pytorch-168c1004b2ca
'''

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils import set_seeds, init_weights_apply
from ResNet18 import LearningLoss


class TrainDDP():
    def __init__(self, gpu_id, params):
        
        self.gpu_id = gpu_id
        self.LL = params['LL']
        self.patience = params['patience'],
        self.model = DDP(params['model'], device_ids=[self.gpu_id], output_device=self.gpu_id, find_unused_parameters=True)
        
        self.dataset_name = params['dataset_name']
        self.method_name = params['method_name']
        
        self.train_dl = params['train_dl']
        self.val_dl = params['val_dl']
        self.test_dl = params['test_dl']
        
        
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.score_fn = params['score_fn']
        self.loss_weird = LearningLoss(self.gpu_id)
        
        self.best_check_filename = f'app/checkpoints/{self.dataset_name}'
        self.init_check_filename = f'{self.best_check_filename}_init_checkpoint.pth.tar'
        
        
        ###########
        set_seeds()
        ###########
        
        print(' => Initializing weights')
        self.model.apply(init_weights_apply)
        print(' DONE\n')
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
    
    
    
    
    def __save_checkpoint(self, filename, check_type):
        print(f' => Saving {check_type} checkpoint')
        checkpoint = {
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, filename)
        print(' DONE\n')
    
    
    
    def __load_checkpoint(self, filename, check_type):
        print(f' => Load {check_type} checkpoint')
        checkpoint = torch.load(filename, map_location={'cuda:%d' % 0: 'cuda:%d' % self.gpu_id})
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(' DONE\n')



    def compute_losses(self, weight, out_weird, outputs, labels, tot_loss_ce, tot_loss_weird):
        
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




    def evaluate(self, dataloader, weight):
                
        tot_loss, tot_loss_ce, tot_loss_weird, tot_accuracy = .0, .0, .0, .0
        
        self.model.eval()

        with torch.inference_mode(): # Allow inference mode
            for _, images, labels in dataloader:
                
                images, labels = images.to(self.gpu_id, non_blocking=True), labels.to(self.gpu_id, non_blocking=True)
                outputs, _, out_weird, _ = self.model(images)

                loss, tot_loss_ce, tot_loss_weird = self.compute_losses(weight, out_weird, outputs, labels, tot_loss_ce, tot_loss_weird)
                
                tot_accuracy += self.score_fn(outputs, labels)
                tot_loss += loss.item()


            tot_accuracy /= len(dataloader)
            tot_loss /= len(dataloader)
            tot_loss_ce /= len(dataloader)
            tot_loss_weird /= len(dataloader)
            
            
        return tot_accuracy, tot_loss, tot_loss_ce, tot_loss_weird
    
    
    
    def train_evaluate(self, epochs):
        
        weight = 1.
    
        check_best_path = f'{self.best_check_filename}/best_{self.method_name}.pth.tar'
		
        best_val_accuracy = 0.0
        
        #results = { 'train_loss': [], 'train_loss_ce': [], 'train_loss_weird': [], 'train_accuracy': [], 
        #            'val_loss': [], 'val_loss_ce': [], 'val_loss_weird': [], 'val_accuracy': [] }
	
 
        for epoch in range(epochs):
                        
            self.model.train()
            
            if epoch == 160:
                print('Decreasing learning rate to 0.01\n')
                for g in self.optimizer.param_groups: g['lr'] = 0.01
            
            if self.LL and epoch == 121:
                print('Ignoring the learning loss form now\n') 
                weight = 0
            
            train_loss, train_loss_ce, train_loss_weird, train_accuracy = .0, .0, .0, .0
            
            self.train_dl.sampler.set_epoch(epoch)
            
            print(f"\n[GPU{self.gpu_id}] Epoch {epoch + 1} | Batchsize: {len(next(iter(self.train_dl))[0])} | Steps: {len(self.train_dl)}")
            
            for _, images, labels in self.train_dl:                
                                    
                images, labels = images.to(self.gpu_id, non_blocking=True), labels.to(self.gpu_id, non_blocking=True)
                self.optimizer.zero_grad()
                outputs, _, out_weird, _ = self.model(images)
                
                loss, _, _  = self.compute_losses(weight, out_weird, outputs, labels, train_loss_ce, train_loss_weird)
                                
                loss.backward()

                self.optimizer.step()
                
                train_loss += loss.item()
                train_accuracy += self.score_fn(outputs, labels)

    

            train_accuracy /= len(self.train_dl)
            train_loss /= len(self.train_dl)
            train_loss_ce /= len(self.train_dl)
            train_loss_weird /= len(self.train_dl)
            
            
            
            ###################################
            # CosineAnnealingLR
            #self.scheduler.step()
            ###################################
            

            '''results['train_loss'].append(train_loss)
            results['train_loss_ce'].append(train_loss_ce)
            results['train_loss_weird'].append(train_loss_weird)
            results['train_accuracy'].append(train_accuracy)'''
            
            # evaluating using the validation set
            val_accuracy, val_loss, val_loss_ce, val_loss_weird = self.evaluate(self.val_dl, weight)
            
            
            '''results['val_loss'].append(val_loss)
            results['val_loss_ce'].append(val_loss_ce)
            results['val_loss_weird'].append(val_loss_weird)
            results['val_accuracy'].append(val_accuracy)'''
            
            
            if(val_accuracy > best_val_accuracy):
                best_val_accuracy = val_accuracy
                
                print('GPU: [{}] | Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, best_val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
                    self.gpu_id, epoch + 1, train_accuracy, train_loss, val_accuracy, best_val_accuracy, val_loss))
                
                if self.gpu_id == 0:
                    self.__save_checkpoint(check_best_path, 'best')
                
            else:
                print('GPU: [{}] | Epoch [{}], train_accuracy: {:.6f}, train_loss: {:.6f}, val_accuracy: {:.6f}, best_val_accuracy: {:.6f}, val_loss: {:.6f} \n'.format(
                   self.gpu_id, epoch + 1, train_accuracy, train_loss, val_accuracy, best_val_accuracy, val_loss))
        
        
        # load best checkpoint
        #self.__load_checkpoint(check_best_path, 'best')
        
        print(f'GPU: {self.gpu_id} | Finished Training\n')
        
        #return {'model_name': self.method_name, 'results': results}




    def test(self):
        test_accuracy, test_loss, test_loss_ce, test_loss_weird = self.evaluate(self.test_dl, weight=1)
        
        if test_loss_ce != 0.0:
            print('TESTING RESULTS -> test_accuracy: {:.6f}, test_loss: {:.6f}, test_loss_ce: {:.6f} , test_loss_weird: {:.6f}\n\n'.format(test_accuracy, test_loss, test_loss_ce, test_loss_weird ))
        else:
            print('TESTING RESULTS -> test_accuracy: {:.6f}, test_loss: {:.6f}\n\n'.format(test_accuracy, test_loss ))
            
            
        return test_accuracy, test_loss, test_loss_ce, test_loss_weird

        
        