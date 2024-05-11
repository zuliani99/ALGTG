import math
import torch.nn as nn

from typing import List

class DeepConvAutoencoder(nn.Module):
    """ Conv Ae with variable number of conv layers """
    def __init__(self, inp_side_len=28, dims: List[int] = [5, 10],
                 kernel_sizes: int | List = 3, central_dim=100, pool=True):
        super().__init__()

        # initial checks
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(dims)
        assert len(kernel_sizes) == len(dims) and all(size > 0 for size in kernel_sizes)

        # build encoder
        step_pool = 1 if len(dims) < 3 else (2 if len(dims) < 6 else 3)
        side_len = inp_side_len
        side_lengths = [side_len]
        dims = [1, *dims]
        enc_layers = []
        for i in range(len(dims) - 1):
            pad = (kernel_sizes[i] - 1) // 2
            enc_layers.append(nn.Conv2d(in_channels=dims[i], out_channels=dims[i + 1], kernel_size=kernel_sizes[i],
                                        padding=pad, stride=1))
            enc_layers.append(nn.ReLU(inplace=True))
            if pool and (i % step_pool == 0 or i == len(dims) - 1) and side_len > 3:
                enc_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                side_len = math.floor(side_len / 2)
                side_lengths.append(side_len)

        # fully connected layers in the center of the autoencoder to reduce dimensionality
        fc_dims = (side_len ** 2 * dims[-1], side_len ** 2 * dims[-1] // 2, central_dim)
        self.encoder = nn.Sequential(
            *enc_layers,
            nn.Flatten(),
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[1], fc_dims[2]),
            nn.ReLU(inplace=True)
        )

        # build decoder
        central_side_len = side_lengths.pop(-1)
        # side_lengths = side_lengths[:-1]
        dec_layers = []
        for i in reversed(range(1, len(dims))):
            # set kernel size, padding and stride to get the correct output shape
            kersize = 2 if len(side_lengths) > 0 and side_len * 2 == side_lengths.pop(-1) else 3
            pad, stride = (1, 1) if side_len == inp_side_len else (0, 2)
            # create transpose convolution layer
            dec_layers.append(nn.ConvTranspose2d(in_channels=dims[i], out_channels=dims[i - 1], kernel_size=kersize,
                                                 padding=pad, stride=stride))
            side_len = side_len if pad == 1 else (side_len * 2 if kersize == 2 else side_len * 2 + 1)
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Sigmoid()

        self.decoder = nn.Sequential(
            nn.Linear(fc_dims[2], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[1], fc_dims[0]),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(dims[-1], central_side_len, central_side_len)),
            *dec_layers,
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded