import torch
import os
import torch.nn as nn
from transformers import AutoModel

class MambaVision(nn.Module):
    '''
    MambaVision Backbone:
        Loading the pretrained models using the name format "MambaVision-<scale>-<dataset>":
            - <scale>: T-Tiny, S-Small, B-Big, L-Large, 
            - <dataset>: ImageNet 21K, ImageNet 1K
        Extracting features in 4 stages of the model
    '''
    def __init__(self, model_name: str = "nvidia/MambaVision-T-1K", pretrained: bool = True):
        '''
        Args:
            model_name (str): MambaVision model's name
            pretrained (bool): Usage of pretrained-model            
        '''
        super().__init__()

        # This follows the instruction from HuggingFace https://huggingface.co/nvidia/MambaVision-T-1K
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code = True
        )

    def forward(self,x):
        '''
        Return the extracted features from the input x in every 4 stages 
            Stage 1: [batch_size, C, H/4, W/4]
            Stage 2: [batch_size, 2C, H/8, W/8]
            Stage 3: [batch_size, 4C, H/16, W/16]
            Stage 4: [batch_size, 8C, H/32, W/32]
        '''
        out_avg_pool, features = self.model(x)

        return features

# ------------------------------
# BACKBONE TESTING
# ------------------------------     
from PIL import Image
from timm.data.transforms_factory import create_transform
from src.utils.plot_feature_map import plot_feature_map

def test_backbone(model_name = "nvidia/MambaVision-T-1K", pretrained = True):
    '''
    Script for testing MambaVision backbone loading
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MambaVision(model_name = model_name, pretrained = pretrained).to(device)
    model.eval()

    img_path = "data/bear.jpeg"
    image = Image.open(img_path).convert("RGB")

    transform = create_transform(
    input_size=(3, 224, 224),
    is_training=False,
    mean=model.model.config.mean,
    std=model.model.config.std,
    crop_mode=model.model.config.crop_mode,
    crop_pct=model.model.config.crop_pct,
    )

    x = transform(image).unsqueeze(0).to(device)

    # Fowarding
    with torch.no_grad():
        features = model(x)

    # Print the feature size
    print("Input shape:", x.shape)

    for i, f in enumerate(features):
        print(f"Stage {i+1}: {f.shape}")
        plot_feature_map(f)

if __name__ == '__main__':
    test_backbone(model_name = "nvidia/MambaVision-T-1K", pretrained = True)