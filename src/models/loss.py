import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.match_locations import match_locations
from src.utils.flatten_predictions import flatten_predictions
from src.utils.generate_locations import generate_locations


# -------------------------
# SIGMOID FOCAL LOSS (NO BACKGROUND CLASS)
# -------------------------
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    logits: [N, C]
    targets: [N] (class index, -1 for neg)
    """
    N, C = logits.shape

    # one-hot (no background class)
    targets_onehot = torch.zeros_like(logits)
    pos_mask = targets >= 0
    targets_onehot[pos_mask, targets[pos_mask]] = 1.0

    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction='none')

    p_t = prob * targets_onehot + (1 - prob) * (1 - targets_onehot)
    loss = ce * ((1 - p_t) ** gamma)

    alpha_t = alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot)
    loss = alpha_t * loss

    return loss.sum()


# -------------------------
# IOU LOSS (FCOS LTRB)
# -------------------------
def iou_loss(pred, target):
    eps = 1e-6

    inter_w = torch.min(pred[:, 0], target[:, 0]) + torch.min(pred[:, 2], target[:, 2])
    inter_h = torch.min(pred[:, 1], target[:, 1]) + torch.min(pred[:, 3], target[:, 3])

    inter = inter_w.clamp(min=0) * inter_h.clamp(min=0)

    area_p = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
    area_t = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])

    union = area_p + area_t - inter
    iou = inter / (union + eps)

    return (1 - iou).sum()


# -------------------------
# MAIN LOSS
# -------------------------
class Model_Loss(nn.Module):
    def __init__(self, num_classes=80, alpha=1.0, beta=1.0, gamma=6.125, strides=[8,16,32]):
        super().__init__()

        self.num_classes = num_classes  # ❗ bỏ background
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.strides = strides

    def forward(self, outputs, targets):
        cls_preds  = flatten_predictions(outputs["cls"])   # [B, N, C]
        box_preds  = flatten_predictions(outputs["box"])   # [B, N, 4]
        coef_preds = flatten_predictions(outputs["coef"])  # [B, N, P]
        proto      = outputs["proto"]                      # [B, P, H, W]

        locations = generate_locations(outputs["cls"], self.strides).to(cls_preds.device)

        B, N, _ = cls_preds.shape

        all_cls_loss = 0.0
        all_box_loss = 0.0
        all_mask_loss = 0.0
        total_num_pos = 0

        for i in range(B):
            gt_boxes  = targets[i]["boxes"].to(cls_preds.device)
            gt_labels = targets[i]["labels"].to(cls_preds.device)
            gt_masks  = targets[i]["masks"].to(cls_preds.device)

            matched_idx, pos_mask = match_locations(locations, gt_boxes)

            n_pos = pos_mask.sum().item()
            total_num_pos += n_pos

            # -------------------------
            # CLASSIFICATION
            # -------------------------
            gt_cls_target = torch.full((N,), -1, dtype=torch.long, device=cls_preds.device)

            if n_pos > 0:
                gt_cls_target[pos_mask] = gt_labels[matched_idx[pos_mask]]

            all_cls_loss += focal_loss(cls_preds[i], gt_cls_target)

            if n_pos == 0:
                continue

            # -------------------------
            # BOX LOSS
            # -------------------------
            pos_box_preds = box_preds[i][pos_mask]
            pos_locs = locations[pos_mask]
            matched_gt_boxes = gt_boxes[matched_idx[pos_mask]]

            l = pos_locs[:, 0] - matched_gt_boxes[:, 0]
            t = pos_locs[:, 1] - matched_gt_boxes[:, 1]
            r = matched_gt_boxes[:, 2] - pos_locs[:, 0]
            b = matched_gt_boxes[:, 3] - pos_locs[:, 1]

            target_ltrb = torch.stack([l, t, r, b], dim=1).clamp(min=0)

            all_box_loss += iou_loss(pos_box_preds, target_ltrb)

            # -------------------------
            # MASK LOSS
            # -------------------------
            pos_coefs = coef_preds[i][pos_mask]  # [n_pos, P]
            P, p_h, p_w = proto.shape[1:]

            # generate masks
            pred_masks = torch.sigmoid(pos_coefs @ proto[i].view(P, -1))
            pred_masks = pred_masks.view(-1, p_h, p_w)

            # resize GT masks
            target_masks = F.interpolate(
                gt_masks[matched_idx[pos_mask]].unsqueeze(1).float(),
                size=(p_h, p_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            # scale boxes theo GT mask (CHUẨN)
            H_img, W_img = gt_masks.shape[-2:]
            scale_x = p_w / W_img
            scale_y = p_h / H_img

            scaled_boxes = matched_gt_boxes.clone()
            scaled_boxes[:, [0, 2]] *= scale_x
            scaled_boxes[:, [1, 3]] *= scale_y

            # tạo mask crop
            y_coords, x_coords = torch.meshgrid(
                torch.arange(p_h, device=pred_masks.device),
                torch.arange(p_w, device=pred_masks.device),
                indexing='ij'
            )

            y_coords = y_coords.unsqueeze(0)
            x_coords = x_coords.unsqueeze(0)

            box_mask = (
                (x_coords >= scaled_boxes[:, 0].view(-1,1,1)) &
                (x_coords <= scaled_boxes[:, 2].view(-1,1,1)) &
                (y_coords >= scaled_boxes[:, 1].view(-1,1,1)) &
                (y_coords <= scaled_boxes[:, 3].view(-1,1,1))
            )

            mask_bce = F.binary_cross_entropy(pred_masks, target_masks, reduction='none')

            all_mask_loss += (mask_bce * box_mask.float()).sum() / box_mask.float().sum().clamp(min=1.0)

        # -------------------------
        # NORMALIZE
        # -------------------------
        divisor = max(total_num_pos, 1)

        final_loss_cls  = (all_cls_loss / divisor) * self.alpha
        final_loss_box  = (all_box_loss / divisor) * self.beta
        final_loss_mask = (all_mask_loss / divisor) * self.gamma  # ❗ FIXED

        return {
            "loss": final_loss_cls + final_loss_box + final_loss_mask,
            "loss_cls": final_loss_cls,
            "loss_box": final_loss_box,
            "loss_mask": final_loss_mask
        }