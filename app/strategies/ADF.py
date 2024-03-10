
import numpy as np
import torch

from strategies.Strategies import Strategies
from typing import Dict, Any, List



class AdversarialDeepFool(Strategies):
    
    def __init__(self, al_params: Dict[str, Any], LL: bool) -> None:
        super().__init__(al_params, LL)
        
        self.method_name = f'{self.__class__.__name__}_LL' if LL else self.__class__.__name__
        self.args.max_iter = 50
        

    def cal_dis(self, x):
        nx = torch.unsqueeze(x, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        out, _, _, _ = self.model(nx+eta)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.args.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py: continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi

            eta += ri.clone()
            nx.grad.data.zero_()
            out, _, _, _ = self.model(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1

        return (eta*eta).sum()


    def query(self, sample_unlab_subset: List[int], n_top_k_obs: int) -> List[int]:

        checkpoint = torch.load(f'{self.best_check_filename}/best_{self.method_name}_cuda:0.pth.tar', map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.model.cpu()
        self.model.eval()
        
        distances = torch.zeros(len(sample_unlab_subset.indices))

        for i, (_, image, _) in enumerate(sample_unlab_subset):
            distances[i] = self.cal_dis(image)

        self.model.cuda()

        overall_topk = torch.topk(distances, n_top_k_obs)
        
        return [sample_unlab_subset.indices[id] for id in overall_topk.indices.tolist()]