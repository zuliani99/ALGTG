import math
import torch.nn as nn


from models.backbones.ResNet18 import ResNet18

class BackBone_Decoder(nn.Module):
    def __init__(self, classes, n_channels, image_size):
        super().__init__()

        # initial checks
        self.encoder = ResNet18(n_classes=classes, n_channels=n_channels)
        self.image_size = image_size
        

        
        # build decoder
        dims = [512, 256, 128, 64]  # dimensions of each layer in the decoder
        side_len = self.image_size // (2 ** (len(dims) - 1))  # initial side length
        dec_layers = []
        for i in reversed(range(1, len(dims))):
            # set kernel size, padding and stride to get the correct output shape
            kersize = 2 if side_len * 2 == self.image_size // (2 ** (i - 1)) else 3
            pad, stride = (1, 1) if side_len == self.image_size // (2 ** i) else (0, 2)
            # create transpose convolution layer
            dec_layers.append(nn.ConvTranspose2d(in_channels=dims[i], out_channels=dims[i - 1], kernel_size=kersize,
                             padding=pad, stride=stride))
            side_len = side_len if pad == 1 else (side_len * 2 if kersize == 2 else side_len * 2 + 1)
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Sigmoid()

        self.decoder = nn.Sequential(
            nn.Linear(512, dims[0] * side_len * side_len),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(dims[0], side_len, side_len)),
            *dec_layers,
        )
        
    def forward(self, x):
        _, encoded = self.encoder(x) # bs x 512
        decoded = self.decoder(encoded)
        return encoded, decoded