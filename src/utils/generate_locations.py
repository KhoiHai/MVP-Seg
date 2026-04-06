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
        B, C, H, W = f.shape

        shifts_x = torch.arange(0, W * stride, step=stride, device=f.device, dtype=torch.float32) + float(stride // 2)
        shifts_y = torch.arange(0, H * stride, step=stride, device=f.device, dtype=torch.float32) + float(stride // 2)

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        loc = torch.stack((shift_x, shift_y), dim=-1)  # [H, W, 2]
        loc = loc.reshape(-1, 2)

        locations.append(loc)

    return torch.cat(locations, dim=0).to(torch.float32)  # [N, 2]
