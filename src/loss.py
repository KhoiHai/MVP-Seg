import torch
import torch.nn as nn
import torch.nn.functional as F


def box_iou(boxes_a, boxes_b):
    '''
    Compute pairwise IoU between two sets of bounding boxes
        Both inputs must be in pascal_voc format [x1, y1, x2, y2]
    Args:
        boxes_a: [N, 4]
        boxes_b: [M, 4]
    Returns:
        iou: [N, M]
    '''
    # Individual areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]) # [N]
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]) # [M]

    # Intersection coordinates via broadcasting
    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])  # [N, M]
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    # Intersection area (clamp to 0 for non-overlapping boxes)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)    # [N, M]

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def match_predictions(pred_boxes, gt_boxes, iou_threshold = 0.5):
    '''
    Match each predicted box to a ground truth box using IoU
        A prediction is positive if its best matching GT IoU >= iou_threshold
        Otherwise it is treated as background
    Args:
        pred_boxes   : [N, 4] predicted boxes for one image
        gt_boxes     : [M, 4] ground truth boxes for one image
        iou_threshold: Minimum IoU to assign a prediction as positive
    Returns:
        matched_gt_idx: [N] GT index for each prediction (-1 = background)
        pos_mask      : [N] boolean mask, True for positive predictions
    '''
    # No GT boxes -> all predictions are background
    if gt_boxes.shape[0] == 0:
        return (
            torch.full((pred_boxes.shape[0],), -1, dtype = torch.long),
            torch.zeros(pred_boxes.shape[0], dtype = torch.bool)
        )

    iou = box_iou(pred_boxes, gt_boxes)         # [N, M]
    best_gt_iou, best_gt_idx = iou.max(dim = 1) # [N] best matching GT per prediction

    pos_mask = best_gt_iou >= iou_threshold
    matched_gt_idx = best_gt_idx.clone()
    matched_gt_idx[~pos_mask] = -1              # Mark negatives as -1

    return matched_gt_idx, pos_mask


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


class YOLACTLoss(nn.Module):
    '''
    YOLACT-style Loss for Instance Segmentation
        Total loss: L = alpha * L_cls + beta * L_box + gamma * L_mask
        - L_cls  : Cross-Entropy classification loss over all anchors
        - L_box  : Smooth L1 regression loss on positive anchors only
        - L_mask : Binary Cross-Entropy between predicted and GT masks
        Loss weights (alpha, beta, gamma) = (1.0, 1.5, 6.125) follow the YOLACT paper
    '''
    def __init__(self, num_classes = 80, num_prototypes = 32,
                 alpha = 1.0, beta = 1.5, gamma = 6.125):
        '''
        Args:
            num_classes   : Number of foreground classes (80 for COCO)
            num_prototypes: Number of prototype masks P (must match Protonet output)
            alpha         : Weight for classification loss
            beta          : Weight for bounding box regression loss
            gamma         : Weight for mask loss (higher because mask is pixel-level)
        '''
        super().__init__()
        self.num_classes    = num_classes
        self.num_prototypes = num_prototypes
        self.alpha          = alpha
        self.beta           = beta
        self.gamma          = gamma

    def forward(self, outputs, targets):
        '''
        Args:
            outputs: dict from MVPSeg.forward()
                cls   : list of [B, num_classes, Hi, Wi]
                box   : list of [B, 4, Hi, Wi]
                coef  : list of [B, num_prototypes, Hi, Wi]
                proto : [B, num_prototypes, H/4, W/4]
            targets: list of dicts (one per image in batch)
                boxes  : [N, 4]   ground truth boxes
                labels : [N]      ground truth class indices
                masks  : [N, H, W] ground truth binary masks
        Returns:
            dict with keys: loss, loss_cls, loss_box, loss_mask
        '''
        # Flatten all scale outputs: [B, total_anchors, C]
        cls_preds  = flatten_predictions(outputs["cls"])   # [B, #pred, num_classes]
        box_preds  = flatten_predictions(outputs["box"])   # [B, #pred, 4]
        coef_preds = flatten_predictions(outputs["coef"])  # [B, #pred, P]
        proto      = outputs["proto"]                      # [B, P, H/4, W/4]

        B = cls_preds.shape[0]
        total_cls, total_box, total_mask = 0.0, 0.0, 0.0
        num_pos = 0

        for i in range(B):
            gt_boxes  = targets[i]["boxes"].to(cls_preds.device)   # [N, 4]
            gt_labels = targets[i]["labels"].to(cls_preds.device)  # [N]
            gt_masks  = targets[i]["masks"].to(cls_preds.device)   # [N, H, W]

            pred_box  = box_preds[i]    # [#pred, 4]
            pred_cls  = cls_preds[i]    # [#pred, num_classes]
            pred_coef = coef_preds[i]   # [#pred, P]
            proto_i   = proto[i]        # [P, H/4, W/4]

            # Match predictions to GT boxes
            matched_idx, pos_mask = match_predictions(pred_box, gt_boxes)
            n_pos = pos_mask.sum().item()
            if n_pos == 0:
                continue
            num_pos += n_pos

            # --- Classification Loss ---
            # Background anchors have target class = 0 (index beyond foreground)
            gt_cls_target = torch.zeros(
                pred_cls.shape[0], dtype = torch.long, device = pred_cls.device
            )
            gt_cls_target[pos_mask] = gt_labels[matched_idx[pos_mask]]
            total_cls += F.cross_entropy(pred_cls, gt_cls_target, reduction = "sum")

            # --- Bounding Box Regression Loss (Smooth L1, positives only) ---
            pos_pred_box = pred_box[pos_mask]                           # [n_pos, 4]
            pos_gt_box   = gt_boxes[matched_idx[pos_mask]]              # [n_pos, 4]
            total_box   += F.smooth_l1_loss(pos_pred_box, pos_gt_box, reduction = "sum")

            # --- Mask Loss ---
            # Reconstruct masks: M = sigmoid(proto @ coef^T) per YOLACT
            pos_coef   = pred_coef[pos_mask]                            # [n_pos, P]
            Ph, Pw     = proto_i.shape[1], proto_i.shape[2]
            proto_flat = proto_i.permute(1, 2, 0).reshape(-1, proto_i.shape[0])
            # [H/4*W/4, P] @ [P, n_pos] -> [H/4*W/4, n_pos]
            pred_masks = torch.sigmoid(proto_flat @ pos_coef.T)
            pred_masks = pred_masks.reshape(Ph, Pw, -1).permute(2, 0, 1)  # [n_pos, H/4, W/4]

            # Downsample GT masks to match prototype resolution H/4 x W/4
            gt_masks_matched = gt_masks[matched_idx[pos_mask]]          # [n_pos, H, W]
            gt_masks_small   = F.interpolate(
                gt_masks_matched.unsqueeze(1), size = (Ph, Pw), mode = "nearest"
            ).squeeze(1)                                                 # [n_pos, H/4, W/4]

            total_mask += F.binary_cross_entropy(
                pred_masks, gt_masks_small, reduction = "sum"
            )

        # Normalize all losses by total number of positive predictions
        denom     = max(num_pos, 1)
        loss_cls  = self.alpha * total_cls  / denom
        loss_box  = self.beta  * total_box  / denom
        loss_mask = self.gamma * total_mask / denom
        total_loss = loss_cls + loss_box + loss_mask

        return {
            "loss":      total_loss,
            "loss_cls":  loss_cls,
            "loss_box":  loss_box,
            "loss_mask": loss_mask
        }


# ------------------------------
# LOSS TESTING
# ------------------------------
from src.models.mvp_seg import MVPSeg

def test_loss():
    '''
    Script for testing the loss function with dummy data
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy model output (no backbone needed for loss test)
    model = MVPSeg(num_classes = 80, num_prototypes = 32).to(device)
    model.eval()

    # Dummy input image
    x = torch.randn(2, 3, 550, 550).to(device)

    with torch.no_grad():
        outputs = model(x)

    # Dummy ground truth targets (2 images, 3 objects each)
    targets = [
        {
            "boxes":  torch.tensor([[50, 50, 200, 200],
                                    [100, 100, 300, 300]], dtype = torch.float32),
            "labels": torch.tensor([0, 1], dtype = torch.int64),
            "masks":  torch.randint(0, 2, (2, 550, 550), dtype = torch.float32)
        },
        {
            "boxes":  torch.tensor([[30, 30, 150, 150]], dtype = torch.float32),
            "labels": torch.tensor([5], dtype = torch.int64),
            "masks":  torch.randint(0, 2, (1, 550, 550), dtype = torch.float32)
        }
    ]

    criterion = YOLACTLoss(num_classes = 80)
    loss_dict = criterion(outputs, targets)

    print("===== LOSS TEST =====")
    print(f"Total Loss : {loss_dict['loss'].item():.4f}")
    print(f"  Cls Loss : {loss_dict['loss_cls'].item():.4f}")
    print(f"  Box Loss : {loss_dict['loss_box'].item():.4f}")
    print(f"  Mask Loss: {loss_dict['loss_mask'].item():.4f}")

if __name__ == '__main__':
    test_loss()