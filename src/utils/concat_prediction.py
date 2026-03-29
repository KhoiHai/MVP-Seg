import torch

def concat_prediction(preds):
    """
    preds: list of [B, C, H, W]
    return: [B, N, C]
    """
    out = []
    for p in preds:
        B, C, H, W = p.shape
        p = p.permute(0, 2, 3, 1).reshape(B, -1, C)
        out.append(p)
    return torch.cat(out, dim=1)