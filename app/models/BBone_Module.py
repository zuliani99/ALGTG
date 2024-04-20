
import torch
import torch.nn as nn

from models.backbones.ResNet18 import ResNet
from models.backbones.ssd_pytorch.SSD import SSD
from models.backbones.VGG import VGG

from models.modules.GTG_Cls import GTG_Module
from models.modules.LossNet import LossNet

import logging
logger = logging.getLogger(__name__)


class Master_Model(nn.Module):
    def __init__(self, backbone: SSD | ResNet | VGG, added_module: LossNet | GTG_Module | None, dataset_name: str) -> None:
        
        super(Master_Model, self).__init__()
        
        self.backbone = backbone
        self.added_module = added_module
        
        if added_module != None:
            self.name = f'{self.backbone.__class__.__name__}_{self.added_module.__class__.__name__}'
        else: 
            self.name = f'{self.backbone.__class__.__name__}'
            
        
        # backbone and additional module have initialized their respecive layers, so I can save the initial checkpoint
        logger.info(f' => Saving Initial {self.name} checkpoint')
        torch.save(dict(state_dict = self.state_dict()), f'app/checkpoints/{dataset_name}/{self.name}_init.pth.tar')
        logger.info(' DONE\n')
        
        
    def forward(self, x, labels=None, mode='all'):
        if mode == 'all':
            outs, embedds = self.backbone(x)
            if torch.any(torch.isnan(embedds)): print(embedds)
            assert not torch.any(torch.isnan(embedds)), 'embedding is nan'
            assert torch.std(embedds) > 0, 'std is zero or negative'
            
            # module out is:
            # LL -> loss
            # GTG -> loss, mask
            
            if self.added_module != None:
                if self.added_module.__class__.__name__ == 'GTG_Module':
                    #self.added_module.change_pahse('train')
                    #features = self.backbone.get_features()
                    module_out = self.added_module(self.backbone.get_features(), embedds, labels)
                else:
                    module_out = self.added_module(self.backbone.get_features())
                return outs, embedds, module_out
            else:
                return outs, embedds, None
        
        elif mode == 'probs': return self.backbone(x)[0]
        
        elif mode == 'embedds': return self.backbone(x)[1]
        
        elif mode == 'module_out':
            if self.added_module != None:
                _, embedds = self.backbone(x)
                if self.added_module.__class__.__name__ == 'GTG_Module':
                    return self.added_module(self.backbone.get_features(), embedds)
                else: return self.added_module(self.backbone.get_features())
            else:
                raise AttributeError("The Master_Model hasn't got any additional module")

