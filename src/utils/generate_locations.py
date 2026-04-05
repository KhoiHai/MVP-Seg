import torch

def generate_locations(feature_list, strides):
    """
    Generate (px, py) for all predictions across FPN levels
    Args:
        feature_list: list of [B, C, H, W]
        strides: list of stride for each level (e.g., [8,16,32])
    Returns:
        locations: [total_preds, 2]
    """
    locations = []

    for f, stride in zip(feature_list, strides):
        _, _, H, W = f.shape
        stride = float(stride)

        shifts_x = (torch.arange(W, device=f.device, dtype=torch.float32) + 0.5) * stride
        shifts_y = (torch.arange(H, device=f.device, dtype=torch.float32) + 0.5) * stride

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        loc = torch.stack((shift_x, shift_y), dim=-1).reshape(-1, 2)

        locations.append(loc)

    return torch.cat(locations, dim=0)
