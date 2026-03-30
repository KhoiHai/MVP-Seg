import torch

def match_locations(locations, gt_boxes):
    if gt_boxes.shape[0] == 0:
        return (
            torch.full((locations.shape[0],), -1, dtype=torch.long, device=locations.device),
            torch.zeros(locations.shape[0], dtype=torch.bool, device=locations.device)
        )

    xs, ys = locations[:, 0], locations[:, 1]

    l = xs[:, None] - gt_boxes[None, :, 0]
    t = ys[:, None] - gt_boxes[None, :, 1]
    r = gt_boxes[None, :, 2] - xs[:, None]
    b = gt_boxes[None, :, 3] - ys[:, None]

    reg_targets = torch.stack([l, t, r, b], dim=2)  # [N, M, 4]

    inside = reg_targets.min(dim=2).values > 0

    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    areas = areas[None].repeat(locations.shape[0], 1)
    areas[~inside] = float('inf')

    matched_idx = areas.argmin(dim=1)
    pos_mask = inside.any(dim=1)
    matched_idx[~pos_mask] = -1

    return matched_idx, pos_mask