'''
SSD: Single Shot MultiBox Object Detector, in PyTorch

Reference:
    https://github.com/amdegroot/ssd.pytorch
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

#from AL_GTG.app.utils import init_weights_apply
from models.Lossnet import LossNet

from .ssd_layers import *

import logging
logger = logging.getLogger(__name__)




class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    #def __init__(self, loss_net, phase, size, base, extras, head, num_classes, voc_cfg):
    def __init__(self, phase, size, base, extras, head, num_classes, voc_cfg):
        super(SSD, self).__init__()
        #self.loss_net: LossNet = loss_net
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBox(voc_cfg)
        self.priors = self.priorbox.forward().clone().detach()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            #self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.detect = Detect()
            
    def change_phase(self):
        if self.phase == 'train':
            self.softmax = nn.Softmax(dim=-1)
            #self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.detect = Detect()
            self.phase = 'test'
        else:
            del self.softmax
            del self.detect
            self.phase = 'train'
            
    # IF IT GIVES ERROR REGARDING THE DEVICE FOR PRIOR THIS IS THE PROBLEM
    def set_device_priors(self, gpu_id: torch.device): 
        self.device = gpu_id
        self.priors.to(self.device)
            

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        self.features = []
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23): x = self.vgg[k](x)

        self.features.append(x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)): x = self.vgg[k](x)
        sources.append(x)
        self.features.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
                self.features.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                                        loc.view(loc.size(0), -1, 4),  # loc preds
                                        self.softmax(conf.view(conf.size(0), -1, self.num_classes)),  # conf preds
                                        self.priors.to(x.dtype).to(self.device)  # default boxes
                                      )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
                        
        return output, None

    def get_features(self):
        return self.features #[512, 1024, 512, 256, 256, 256]
    
    def get_embedding_dim(self) -> int:
        return 0

    def load_weights(self, base_file):
        _, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            logger.info('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            logger.info('Finished!')
        else:
            logger.info('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


#def build_ssd(loss_net, phase, voc_cfg, size=300, num_classes=21) -> SSD:
#def build_ssd(device, phase, voc_cfg, size=300, num_classes=21) -> SSD:
def build_ssd(phase, voc_cfg, size=300, num_classes=21) -> SSD:
    if phase != "test" and phase != "train":
        logger.exception("ERROR: Phase: " + phase + " not recognized")
        raise Exception("ERROR: Phase: " + phase + " not recognized")
    if size != 300:
        logger.info("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        raise Exception("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    #return SSD(loss_net, phase, size, base_, extras_, head_, num_classes, voc_cfg)
    return SSD(phase, size, base_, extras_, head_, num_classes, voc_cfg)#.to(device)



class SSD_LL(nn.Module):
    #def __init__(self, device, phase, voc_config, num_classes=21, ln_p=None) -> None:
    def __init__(self, phase, voc_config, num_classes=21, ln_p=None) -> None:
        super(SSD_LL, self).__init__()
        self.loss_net = LossNet(ln_p)#.to(device)
        #self.ssd_net = build_ssd(device, phase, voc_config, num_classes=num_classes)
        self.backbone = build_ssd(phase, voc_config, num_classes=num_classes)
        vgg_weights = torch.load('app/models/ssd_pytorch/vgg16_reducedfc.pth')
        self.backbone.vgg.load_state_dict(vgg_weights)
        # initialize
        #self.ssd_net.extras.apply(init_weights_apply)
        #self.ssd_net.loc.apply(init_weights_apply)
        #self.ssd_net.conf.apply(init_weights_apply)
        
    def forward(self, x):
        outs, embedds = self.backbone(x)
        features = self.backbone.get_features()
        pred_loss = self.loss_net(features)
        return outs, embedds, pred_loss


