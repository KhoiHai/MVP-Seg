import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone.mamba_vision import MambaVision
from src.models.head.prediction_head import Prediction_Head
from src.models.head.protonet import Protonet
from src.models.neck.panet_neck import PANet_Neck

class MVP_Seg(nn.Module):
    '''
    Final architecture of MVP_Seg
    '''
    def __init__(self, model_name = "nvidia/MambaVision-T-1K", pretrained = True, 
                 shared_channel = 256, num_classes = 80, num_prototypes = 32):
        '''
        Args: 
            model_name: The name type of MambaVision model
            pretrained: Usage of pretrained
            shared_channel: Number of channel in the shared neck
            num_classes: Number of classes
            num_prototype: Number of prototypes
        '''
        super().__init__()

        # Backbone
        self.backbone = MambaVision(model_name = model_name, pretrained = pretrained)

        # Neck
        self.neck = PANet_Neck(out_channels = shared_channel)

        # Prediction Head
        self.pred_head = Prediction_Head(in_channels = shared_channel, num_classes = num_classes, num_prototypes = num_prototypes)

        # Protonet
        self.proto = Protonet(in_channels = shared_channel, num_prototypes = num_prototypes)
    
    def forward(self, x):
        '''
        Args:
            x: Input image batch [batch_size, 3, H, W]
        Returns:
            dict with keys:
                cls: list of [batch_size, num_classes, Hi, Wi] for each scale
                box: list of [batch_size, 4, Hi, Wi] for each scale
                coef: list of [batch_size, num_prototypes, Hi, Wi] for each scale
                proto: [batch_size, num_prototypes, H/4, W/4]
        '''
        # Backbone
        features = self.backbone(x)
        C2, C3, C4 = features[1], features[2], features[3]

        # Neck
        neck_feature_map = self.neck([C2, C3, C4]) # N2, N3, N4

        # Prediction (All N2, N3, N4 process)
        cls_out, box_out, coef_out = self.pred_head(neck_feature_map)

        coef_out = [torch.tanh(c) for c in coef_out]

        # Prototype (Only N2 process)
        proto_out = torch.sigmoid(self.proto(neck_feature_map[0]))

        return{
            "cls": cls_out,
            "box": box_out,
            "coef": coef_out,
            "proto": proto_out
        }