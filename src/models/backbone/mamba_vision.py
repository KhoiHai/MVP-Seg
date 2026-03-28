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

        # This follows the instruction from HuggingFace https://huggingface.co/nvidia/MambaVision-S-1K
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code = True
        )

    def forward(self,x):
        '''
        Return the extracted features from the input x in every 4 stages 
        '''
        out_avg_pool, features = self.model(x)

        return features
    
if __name__ == "__main__":
    from PIL import Image
    from timm.data.transforms_factory import create_transform

    '''
    Script for testing MambaVision backbone loading
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MambaVision("nvidia/MambaVision-T-1K", pretrained = True).to(device)
    model.eval()

    img_path = "./data/bear.jpeg"
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

    # ===== Forward =====
    with torch.no_grad():
        features = model(x)

    # ===== Print result =====
    print("Input shape:", x.shape)

    for i, f in enumerate(features):
        print(f"Stage {i+1}: {f.shape}")