import torch
import torch.nn.functional as F

# Classification Loss: Cross Entropy
def cls_loss(pred, target):
    """
    pred: [B, N, num_classes]
    target: [B, N]
    """
    return F.cross_entropy(pred.transpose(1, 2), target, reduction='mean')

# Mask Loss: Binary Cross Entropy
def mask_loss(pred_mask, gt_mask):
    """
    pred_mask: [B, N, H, W]
    gt_mask: [B, N, H, W]
    """
    return F.binary_cross_entropy(pred_mask, gt_mask)

# Bounding Box Loss: CIOU Loss
def bbox_iou_ciou(box1, box2):
    """
    box1, box2: [N,4] in x1,y1,x2,y2
    """
    x1 = box1[:,0]; y1 = box1[:,1]; x2 = box1[:,2]; y2 = box1[:,3]
    x1g = box2[:,0]; y1g = box2[:,1]; x2g = box2[:,2]; y2g = box2[:,3]

    # Intersection
    xi1 = torch.max(x1, x1g)
    yi1 = torch.max(y1, y1g)
    xi2 = torch.min(x2, x2g)
    yi2 = torch.min(y2, y2g)
    inter = (xi2 - xi1).clamp(0) * (yi2 - yi1).clamp(0)

    # Union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - inter + 1e-7

    iou = inter / union

    # Center distance
    cx1 = (x1 + x2)/2; cy1 = (y1 + y2)/2
    cx2 = (x1g + x2g)/2; cy2 = (y1g + y2g)/2
    rho2 = (cx1 - cx2)**2 + (cy1 - cy2)**2

    # Convex diagonal
    c2 = (torch.max(x2, x2g) - torch.min(x1, x1g))**2 + \
         (torch.max(y2, y2g) - torch.min(y1, y1g))**2 + 1e-7

    # Aspect ratio term
    w1 = x2 - x1; h1 = y2 - y1
    w2 = x2g - x1g; h2 = y2g - y1g
    v = (4 / (3.14159265 **2)) * (torch.atan(w2/h2) - torch.atan(w1/h1))**2
    alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (rho2/c2) - alpha*v
    return ciou

def ciou_loss(pred, target):
    """
    pred, target: [N,4] x1,y1,x2,y2
    """
    ciou = bbox_iou_ciou(pred, target)
    return (1 - ciou).mean()