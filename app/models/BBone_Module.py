

import torch.nn as nn

from app.models.backbones.ResNet18 import ResNet
from app.models.backbones.ssd_pytorch.SSD import SSD
from app.models.modules.GTG_Cls import GTG_Module
from app.models.modules.LossNet import LossNet

class Master_Model(nn.Module):
    def __init__(self, backbone: SSD | ResNet, module: LossNet | GTG_Module | None) -> None:
        super(Master_Model, self).__init__()
        self.backbone = backbone
        self.module = module
        self.name = f'{self.backbone.__class__.__name__}_{self.module.__class__.__name__}'
        
    def forward(self, x, mode='all'):
        if mode == 'all':
            outs, embedds = self.backbone(x)
            if self.module != None:
                module_out = self.module(self.backbone.get_features())
                # module out is:
                # LL -> loss
                # GTG -> loss, mask
                return outs, embedds, module_out
            else:
                return outs, embedds
        elif mode == 'probs':
            outs, _ = self.backbone(x)
            return outs
        elif mode == 'embedds':
            _, embedds = self.backbone(x)
            return embedds
        elif mode == 'module_out':
            if self.module != None:
                _, _ = self.backbone(x)
                return self.module(self.backbone.get_features())
            else:
                raise AttributeError("The Master_Model hasn't got any additional module")

