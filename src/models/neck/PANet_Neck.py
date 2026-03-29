import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormAct(nn.Module):
    '''
    ConvNormAct Block 
    Conv -> Normalization Layer -> Activation
    '''
    def __init__(self, in_c, out_c, k, s = 1, p = 0, norm = "bn", act = "relu"):
        '''
        Args:
            in_c: Channel of input
            out_c: Channel of output
            k: Kernel size
            s: Stride
            p: Padding
            norm: The normalization layer used includes BatchNorm and GroupNorm
            act: The activation function used includes ReLU and SiLU
        '''

        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, k, stride = s, padding = p, bias = False)

        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_c)
        elif norm == "gn":
            self.norm = nn.GroupNorm(32, out_c)

        if act == "relu":
            self.act = nn.ReLU(inplace = True)
        elif act == "silu":
            self.act = nn.SiLU()
        else: 
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class PANet_Neck(nn.Module):
    def __init__(self, in_channels = [160, 320, 640], out_channels = 256, norm = "bn", act = "relu"):
        '''
        in_channels : Input channels of C2, C3, C4 from stage 2, 3, 4 of the backbone
        out_channels: The out channels C_new which is 256 bys default
        norm: Normalization layer used
        act: Activation function used
        '''
        super().__init__()

        # Lateral (Make sure the feature map has the same channels for addition)
        self.lateral = nn.ModuleList([
            ConvNormAct(in_channels[0], out_channels, 1, norm = norm, act = act), # L2 (C_new, H/8, W/8) using 1x1 kernel
            ConvNormAct(in_channels[1], out_channels, 1, norm = norm, act = act), # L3 (C_new, H/16, W/16) using 1x1 kernel
            ConvNormAct(in_channels[2], out_channels, 1, norm = norm, act = act) # L4 (C_new, H/32, W/32) using 1x1 kernel
        ])

        # Top down refine (smoothing after upsample with interpolation)
        self.refine_top_down = nn.ModuleList([
            ConvNormAct(out_channels, out_channels, 3, p = 1, norm = norm, act = act), # P2
            ConvNormAct(out_channels, out_channels, 3, p = 1, norm = norm, act = act), # P3
            ConvNormAct(out_channels, out_channels, 3, p = 1, norm = norm, act = act) # P4
        ])

        # Bottom up
        self.downsample = nn.ModuleList([
            ConvNormAct(out_channels, out_channels, 3, s=2, p=1, norm=norm, act=act),
            ConvNormAct(out_channels, out_channels, 3, s=2, p=1, norm=norm, act=act),
        ])

        # Bottom up refine
        self.refine_bottom_up = nn.ModuleList([
            ConvNormAct(out_channels, out_channels, 3, p=1, norm=norm, act=act),  # N3
            ConvNormAct(out_channels, out_channels, 3, p=1, norm=norm, act=act),  # N4
        ])

    def forward(self, features):
        '''
        Take the 3 features: C2, C3, C4 from stage 2, 3, 4 of the backbone
        '''
        C2, C3, C4 = features

        # Leteral Phase
        L2 = self.lateral[0](C2)
        L3 = self.lateral[1](C3)
        L4 = self.lateral[2](C4)

        # Top-down Phase
        P4 = L4 

        P3 = L3 + F.interpolate(P4, size = L3.shape[-2:], mode = "nearest")
        P3 = self.refine_top_down[1](P3)

        P2 = L2 + F.interpolate(P3, size = L2.shape[-2:], mode = "nearest")
        P2 = self.refine_top_down[0](P2)

        P4 = self.refine_top_down[2](P4)

        # Bottom-up Phase
        N2 = P2

        N3 = P3 + self.downsample[0](N2)
        N3 = self.refine_bottom_up[0](N3)

        N4 = P4 + self.downsample[1](N3)
        N4 = self.refine_bottom_up[1](N4)

        return [N2, N3, N4]
    
# ------------------------------
# NECK TESTING
# ------------------------------   
from src.models.backbone.mamba_vision import MambaVision
from PIL import Image
from timm.data.transforms_factory import create_transform

def test_neck(model_name="nvidia/MambaVision-T-1K", pretrained=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Backbone
    backbone = MambaVision(model_name=model_name, pretrained=pretrained).to(device)
    backbone.eval()

    # Neck
    neck = PANet_Neck(
        in_channels=[160, 320, 640],
        out_channels=256
    ).to(device)
    neck.eval()

    # Image
    img_path = "/content/MVP-Seg/data/bear.jpeg"
    image = Image.open(img_path).convert("RGB")

    transform = create_transform(
        input_size=(3, 224, 224),
        is_training=False,
        mean=backbone.model.config.mean,
        std=backbone.model.config.std,
        crop_mode=backbone.model.config.crop_mode,
        crop_pct=backbone.model.config.crop_pct,
    )

    x = transform(image).unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        features = backbone(x)

    print("\n===== BACKBONE =====")
    for i, f in enumerate(features):
        print(f"Stage {i+1}: {f.shape}")

    # BỎ stage 1
    C2, C3, C4 = features[1], features[2], features[3]

    # Forward neck
    with torch.no_grad():
        outputs = neck([C2, C3, C4])

    print("\n===== NECK =====")
    for i, o in enumerate(outputs):
        print(f"N{i+2}: {o.shape}")