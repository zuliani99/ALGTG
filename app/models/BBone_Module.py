

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





'''class ResNet_LL(nn.Module):
    def __init__(self, image_size: int, n_classes=10,  n_channels=3) -> None:
        super(ResNet_LL, self).__init__()
        self.loss_net = LossNet()
        self.backbone = ResNet18(image_size, n_classes=n_classes, n_channels=n_channels)
        
        
    def forward(self, x, mode='all'):
        if mode == 'all':
            outs, embedds = self.backbone(x)
            pred_loss = self.loss_net(self.backbone.get_features())
            return outs, embedds, pred_loss
        elif mode == 'probs':
            outs, _ = self.backbone(x)
            return outs
        elif mode == 'embedds':
            _, embedds = self.backbone(x)
            return embedds
        elif mode == 'pred_loss':
            _, _ = self.backbone(x)
            return self.loss_net(self.backbone.get_features())
        else: 
            raise AttributeError('You have specified wrong output to return for ResNet_LL')

            
class SSD_LL(nn.Module):
    def __init__(self, phase, voc_config, num_classes=21, ln_p=None) -> None:
        super(SSD_LL, self).__init__()
        self.loss_net = LossNet(ln_p)
        self.backbone = build_ssd(phase, voc_config, num_classes=num_classes)
        vgg_weights = torch.load('app/models/ssd_pytorch/vgg16_reducedfc.pth')
        self.backbone.vgg.load_state_dict(vgg_weights)
        

        
    def forward(self, x, mode='all'):
        if mode == 'all':
            outs, embedds = self.backbone(x)
            pred_loss = self.loss_net(self.backbone.get_features())
            return outs, embedds, pred_loss
        elif mode == 'probs':
            outs, _ = self.backbone(x)
            return outs
        elif mode == 'embedds':
            _, embedds = self.backbone(x)
            return embedds
        elif mode == 'pred_loss':
            _, _ = self.backbone(x)
            return self.loss_net(self.backbone.get_features())
        else: 
            raise AttributeError('You have specified wrong output to return for SSD_LL')



class Class_GTG(nn.Module):
    def __init__(self, gtg_module_params, resnet_params) -> None:
    
        super(Class_GTG, self).__init__()
        self.gtg = GTG_Module(gtg_module_params)
        self.backbone = ResNet18(
            image_size=resnet_params['image_size'],
            n_classes=resnet_params['n_classes'],
            n_channels=resnet_params['n_channels']
        )
        
    def forward(self, images, labels, mode='all'):
        if mode == 'all':
            outs, embedds = self.backbone(images)
            y_pred, y_true, mask = self.gtg(embedds, labels)
            return outs, embedds, nn.MSELoss(y_pred, y_true), mask 
        elif mode == 'probs':
            outs, _ = self.backbone(images)
            return outs
        elif mode == 'embedds':
            _, embedds = self.backbone(images)
            return embedds
        elif mode == 'pred_GTG':
            outs, embedds = self.backbone(images)
            self.gtg(embedds, labels)
        else: 
            raise AttributeError('You have specified wrong output to return for ResNet_LL')
    
'''