
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from strategies.Strategies import Strategies, Subset
from utils import entropy

from typing import Dict, Any, List, Tuple





class BALD(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], training_params: Dict[str, Any], LL: bool) -> None:
        super().__init__(al_params, training_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        
        
        
    def evaluate_unlabeled_train(self, n_drop=5) -> Tuple[torch.Tensor, torch.Tensor]:
        
        checkpoint = torch.load(f'{self.best_check_filename}/best_{self.method_name}_cuda:0.pth.tar', map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.model.train()
        
        prob_dist_drop = torch.zeros((n_drop, len(self.unlab_train_dl.dataset), self.n_classes), dtype=torch.float32, device=self.device) 
        indices = torch.empty(0, dtype=torch.int8, device=self.device) 
        
        for drop in range(n_drop):
            with torch.inference_mode(): # Allow inference mode
                for idx_dl, (idxs, images, _) in enumerate(self.unlab_train_dl):
                    
                    idxs, images = idxs.to(self.device), images.to(self.device)
                    
                    outputs, _, _, _ = self.model(images)
                                                         
                    prob_dist_drop[drop][idx_dl * idxs.shape[0] : (idx_dl + 1) * idxs.shape[0]] += F.softmax(outputs, dim=1)

                    if(drop == 0): indices = torch.cat((indices, idxs), dim = 0)
                    
        return indices, prob_dist_drop 
        
    

    def disagreement_dropout(self) -> Tuple[torch.Tensor, torch.Tensor]:
        indices, prob_dist_drop = self.evaluate_unlabeled_train()
        
        mean_pb = torch.mean(prob_dist_drop, dim=0)
        
        entropy1 = entropy(mean_pb)
        entropy2 = torch.mean(entropy(prob_dist_drop, dim=2), dim=0)
        
        del prob_dist_drop
        torch.cuda.empty_cache()
        
        return indices, entropy2 - entropy1
        
    
    
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> List[int]:
                        
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True
        )
            
        print(' => Performing the disagreement dropout')
        indices, res_entropy = self.disagreement_dropout()
        print(' DONE\n')
            
        overall_topk = torch.topk(res_entropy, n_top_k_obs)
        
        del res_entropy
        torch.cuda.empty_cache()
        
        return [indices[id].item() for id in overall_topk.indices.tolist()]
    