import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neck.panet_neck import ConvNormAct

class Prediction_Head(nn.Module):
    '''
    Prediction Head output the classification, bounding box and coefficients
    '''
    def __init__(self, in_channels = 256, num_classes = 80, num_prototypes = 32, num_convs = 2,
                 norm = "bn", act = "silu"):
        '''
        in_channels : The shared number of channels from all feature map extracted from Neck
        num_classes: Number of classes for classification
        num_prototypes: Number of prototypes which is also number of coefficient we need to generate
        num_convs: Number of shared convolution layer
        norm: Normalization layer
        act: Activation function
        '''
        super().__init__()

        shared = []
        for _ in range(num_convs):
            shared.append(ConvNormAct(in_channels, in_channels, 3, p=1, norm=norm, act=act))
        self.shared_conv = nn.Sequential(*shared)

        # Branches
        self.cls_head = nn.Conv2d(in_channels, num_classes, 1)
        self.box_head = nn.Conv2d(in_channels, 4, 1)
        self.coef_head = nn.Conv2d(in_channels, num_prototypes, 1)

    def forward(self, features):
        '''
        Every feature N2, N3, N4 extracted from PANet will be processed in shared convolution layers and split in 3
        sub-branches for classification, bounding box detection and coefficient of prototype
        '''
        cls_outputs = []
        box_outputs = []
        coef_outputs = []

        for f in features:
            x = self.shared_conv(f)

            cls = self.cls_head(x)
            box = F.softplus(self.box_head(x))
            coef = self.coef_head(x)

            cls_outputs.append(cls)
            box_outputs.append(box)
            coef_outputs.append(coef)

        return cls_outputs, box_outputs, coef_outputs