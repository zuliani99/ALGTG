

import torch.nn as nn

from models.backbones.ResNet18 import ResNet
from models.backbones.ssd_pytorch.SSD import SSD
from models.modules.GTG_Cls import GTG_Module
from models.modules.LossNet import LossNet

class Master_Model(nn.Module):
    def __init__(self, backbone: SSD | ResNet, added_module: LossNet | GTG_Module | None) -> None:
        super(Master_Model, self).__init__()
        self.backbone = backbone
        self.added_module = added_module
        if added_module != None:
            self.name = f'{self.backbone.__class__.__name__}_{self.added_module.__class__.__name__}'
        else: 
            self.name = f'{self.backbone.__class__.__name__}'
        
    def forward(self, x, labels, mode='all'):
        if mode == 'all':
            outs, embedds = self.backbone(x)
            if self.added_module != None:
                if self.added_module.__class__.__name__ == 'GTG_Module':
                    self.added_module.change_pahse('train')
                    #module_out = self.added_module(self.backbone.get_features(), labels)
                    module_out = self.added_module(embedds, labels) # intially we take take only the embeddinfg as the last layer before the classification one
                else:
                    module_out = self.added_module(self.backbone.get_features())
                # module out is:
                # LL -> loss
                # GTG -> loss, mask
                return outs, embedds, module_out
            else:
                return outs, embedds, None
        elif mode == 'probs':
            outs, _ = self.backbone(x)
            return outs
        elif mode == 'embedds':
            _, embedds = self.backbone(x)
            return embedds
        elif mode == 'module_out':
            if self.added_module != None:
                _, embedds = self.backbone(x)
                if self.added_module.__class__.__name__ == 'GTG_Module':
                    self.added_module.change_pahse('test')
                    return self.added_module(embedds)
                else:
                    return self.added_module(self.backbone.get_features())
            else:
                raise AttributeError("The Master_Model hasn't got any additional module")

