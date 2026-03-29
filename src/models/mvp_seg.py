import torch
import torch.nn as nn

from src.models.backbone.mamba_vision import MambaVision
from src.models.neck.PANet_Neck import PANet_Neck
from src.models.head.protonet import Protonet
from src.models.head.prediction_head import Prediction_Head

class MVPSeg(nn.Module):
    '''
    MVPSeg: Real-time Instance Segmentation with MambaVision Backbone
        Full pipeline connecting all components:
            Backbone  -> extracts multi-scale features C1, C2, C3, C4
            PANet Neck -> fuses features into N2, N3, N4 (same channels)
            Protonet  -> generates P shared prototype masks from N2
            Pred Head -> predicts cls, box, coef from N2, N3, N4
    '''
    def __init__(self,
                 model_name = "nvidia/MambaVision-T-1K",
                 num_classes = 80,
                 num_prototypes = 32,
                 out_channels = 256):
        '''
        Args:
            model_name: HuggingFace name of MambaVision backbone
            num_classes: Number of object classes (80 for COCO)
            num_prototypes: Number of prototype masks P (32 by default per YOLACT)
            out_channels: Unified channel count C_new from PANet (256 by default)
        '''
        super().__init__()

        # Backbone: MambaVision-T produces channels [80, 160, 320, 640] across 4 stages
        self.backbone = MambaVision(model_name = model_name)

        # Neck: takes C2, C3, C4 (skip C1 which is too coarse), unifies to out_channels
        self.neck = PANet_Neck(
            in_channels = [160, 320, 640],  # C2, C3, C4 of MambaVision-T
            out_channels = out_channels
        )

        # Protonet: generates prototype masks from N2 (highest resolution)
        self.protonet = Protonet(
            in_channels = out_channels,
            num_protypes = num_prototypes
        )

        # Prediction Head: shared across N2, N3, N4 for cls, box and coef prediction
        self.pred_head = Prediction_Head(
            in_channels = out_channels,
            num_classes = num_classes,
            num_prototypes = num_prototypes
        )

    def forward(self, x):
        '''
        Args:
            x: Input image batch [batch_size, 3, H, W]
        Returns:
            dict with keys:
                cls   : list of [batch_size, num_classes, Hi, Wi] for each scale
                box   : list of [batch_size, 4, Hi, Wi] for each scale
                coef  : list of [batch_size, num_prototypes, Hi, Wi] for each scale
                proto : [batch_size, num_prototypes, H/4, W/4]
        '''
        # Stage 1 - Backbone: extract hierarchical features
        features = self.backbone(x)                             # [C1, C2, C3, C4]
        C2, C3, C4 = features[1], features[2], features[3]     # Drop C1 (H/4, too coarse for neck)

        # Stage 2 - Neck: top-down then bottom-up feature fusion
        neck_outs = self.neck([C2, C3, C4])                     # [N2, N3, N4]
        N2, N3, N4 = neck_outs

        # Stage 3 - Protonet: generate shared prototype masks from finest feature N2
        prototypes = self.protonet(N2)                          # [B, P, H/4, W/4]

        # Stage 4 - Prediction Head: classify, regress box and predict coef per scale
        cls_outs, box_outs, coef_outs = self.pred_head([N2, N3, N4])

        return {
            "cls":   cls_outs,    # list of [B, num_classes, Hi, Wi]
            "box":   box_outs,    # list of [B, 4, Hi, Wi]
            "coef":  coef_outs,   # list of [B, P, Hi, Wi]
            "proto": prototypes   # [B, P, H/4, W/4]
        }


# ------------------------------
# MVP-SEG TESTING
# ------------------------------
from PIL import Image
from timm.data.transforms_factory import create_transform

def test_mvpseg(model_name = "nvidia/MambaVision-T-1K", pretrained = True):
    '''
    Script for testing the full MVPSeg pipeline end-to-end
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MVPSeg(
        model_name = model_name,
        num_classes = 80,
        num_prototypes = 32,
        out_channels = 256
    ).to(device)
    model.eval()

    # Image
    img_path = "data/bear.jpeg"
    image = Image.open(img_path).convert("RGB")

    transform = create_transform(
        input_size = (3, 224, 224),
        is_training = False,
        mean = model.backbone.model.config.mean,
        std = model.backbone.model.config.std,
        crop_mode = model.backbone.model.config.crop_mode,
        crop_pct = model.backbone.model.config.crop_pct,
    )

    x = transform(image).unsqueeze(0).to(device)

    # Forward full pipeline
    with torch.no_grad():
        outputs = model(x)

    print("Input shape:", x.shape)
    print("\n===== PREDICTION HEAD OUTPUTS =====")
    for i, (cls, box, coef) in enumerate(zip(outputs["cls"], outputs["box"], outputs["coef"])):
        print(f"Scale N{i+2}: cls={cls.shape}  box={box.shape}  coef={coef.shape}")
    print(f"\nPrototypes: {outputs['proto'].shape}")

if __name__ == '__main__':
    test_mvpseg(model_name = "nvidia/MambaVision-T-1K", pretrained = True)