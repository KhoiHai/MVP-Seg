import torch
import torch.nn.functional as F

def generate_masks(proto, coef):
    """
    proto: [B, K, Hp, Wp]
    coef: [B, N, K]
    return: [B, N, Hp, Wp]
    """
    B, K, Hp, Wp = proto.shape
    proto_flat = proto.view(B, K, -1)          # [B, K, H*W]
    masks = torch.matmul(coef, proto_flat)     # [B, N, H*W]
    masks = masks.view(B, -1, Hp, Wp)
    return torch.sigmoid(masks)