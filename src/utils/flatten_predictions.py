import torch

def flatten_predictions(feature_list):
    '''
    Flatten and concatenate multi-scale prediction tensors
        Converts list of [B, C, Hi, Wi] -> [B, sum(Hi*Wi), C]
    Args:
        feature_list: list of Tensors [B, C, Hi, Wi] from each FPN scale
    Returns:
        Tensor [B, total_anchors, C]
    '''
    out = []
    for f in feature_list:
        B, C, H, W = f.shape
        f = f.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, H*W, C]
        out.append(f)
    return torch.cat(out, dim = 1)                    # [B, sum(H*W), C]