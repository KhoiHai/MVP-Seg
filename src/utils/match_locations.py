import torch

def match_locations(locations, gt_boxes, radius=2.5):
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

    reg_targets = torch.stack([l, t, r, b], dim=2)

    # 🔥 condition 1: inside box
    inside_box = reg_targets.min(dim=2).values > 0

    # 🔥 condition 2: center sampling (QUAN TRỌNG)
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

    # radius theo pixel (scale theo stride)
    radius = radius * 8  # bạn có stride nhỏ nhất = 8

    center_l = xs[:, None] - (cx[None, :] - radius)
    center_t = ys[:, None] - (cy[None, :] - radius)
    center_r = (cx[None, :] + radius) - xs[:, None]
    center_b = (cy[None, :] + radius) - ys[:, None]

    center_box = torch.stack([center_l, center_t, center_r, center_b], dim=2)
    inside_center = center_box.min(dim=2).values > 0

    # 🔥 final mask
    final_mask = inside_box & inside_center

    # assign GT nhỏ nhất
    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    areas = areas[None].repeat(locations.shape[0], 1)
    areas[~final_mask] = float('inf')

    matched_idx = areas.argmin(dim=1)
    pos_mask = final_mask.any(dim=1)
    matched_idx[~pos_mask] = -1

    return matched_idx, pos_mask