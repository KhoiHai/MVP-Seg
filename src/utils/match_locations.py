import torch

def match_locations(locations, gt_boxes, center_radius=2.5, strides=None, level_sizes=None):
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
    inside_box  = reg_targets.min(dim=2).values > 0

    # Center sampling: tránh object nhỏ không có positive location
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    if strides is not None and level_sizes is not None:
        per_loc_radius = torch.empty(locations.shape[0], device=locations.device, dtype=locations.dtype)
        start = 0
        for lvl_size, stride in zip(level_sizes, strides):
            end = start + int(lvl_size)
            per_loc_radius[start:end] = center_radius * float(stride)
            start = end
        if start != locations.shape[0]:
            raise ValueError("Sum(level_sizes) must equal number of locations.")
    else:
        per_loc_radius = torch.full(
            (locations.shape[0],),
            center_radius * 8.0,
            device=locations.device,
            dtype=locations.dtype,
        )

    inside_center = (
        (xs[:, None] - cx[None, :]).abs() <= per_loc_radius[:, None]
    ) & (
        (ys[:, None] - cy[None, :]).abs() <= per_loc_radius[:, None]
    )

    # Positive = trong box VÀ gần center
    inside = inside_box & inside_center

    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    areas = areas[None].repeat(locations.shape[0], 1)
    areas[~inside] = float('inf')

    matched_idx = areas.argmin(dim=1)
    pos_mask    = inside.any(dim=1)
    matched_idx[~pos_mask] = -1

    return matched_idx, pos_mask
