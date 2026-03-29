import torch

def generate_points(feat, stride):
    B, C, H, W = feat.shape
    shifts_x = torch.arange(0, W*stride, step=stride, device=feat.device)
    shifts_y = torch.arange(0, H*stride, step=stride, device=feat.device)
    y, x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    points = torch.stack((x, y), dim=-1)   # [H, W, 2]
    return points.view(-1, 2)               # [H*W, 2]
