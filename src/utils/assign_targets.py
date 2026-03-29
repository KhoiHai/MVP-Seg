import torch

def assign_targets(points, gt_boxes, gt_labels):
    """
    points: [N, 2], gt_boxes: [M,4], gt_labels: [M]
    return:
        cls_target: [N]
        box_target: [N, 4]
    """
    N = points.shape[0]
    cls_target = torch.zeros(N, dtype=torch.long, device=points.device)
    box_target = torch.zeros(N, 4, device=points.device)

    for i in range(gt_boxes.shape[0]):
        x1, y1, x2, y2 = gt_boxes[i]
        l = points[:,0] - x1
        t = points[:,1] - y1
        r = x2 - points[:,0]
        b = y2 - points[:,1]
        inside = (l>0) & (t>0) & (r>0) & (b>0)
        cls_target[inside] = gt_labels[i]
        box_target[inside] = torch.stack([l,t,r,b], dim=1)[inside]
    
    return cls_target, box_target