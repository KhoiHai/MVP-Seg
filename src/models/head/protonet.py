import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neck.panet_neck import ConvNormAct

class Protonet(nn.Module):
    '''
    Protonet for prototype generation
    '''
    def __init__(self, in_channels = 256, norm = "bn", act = "relu", num_protypes = 32):
        '''
        in_channels: The channel of N2 feature map
        norm: Normalization layer
        act: Activation function used
        num_prototypes: Number of prototypes generate
        '''
        super().__init__()

        self.proto = nn.Sequential(
            ConvNormAct(in_channels, 256, 3, p = 1),
            ConvNormAct(256, 256, 3, p = 1),
            ConvNormAct(256, 256, 3, p = 1),

            nn.Upsample(scale_factor = 2, mode = "nearest"),

            ConvNormAct(256, 256, 3, p = 1),
            ConvNormAct(256, 256, 3, p = 1),

            nn.Conv2d(256, num_protypes, 1)
        )

    def forward(self, x):
        x = self.prot(x)
        return x