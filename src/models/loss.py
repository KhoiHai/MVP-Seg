import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.match_locations import match_locations
from src.utils.flatten_predictions import flatten_predictions
from src.utils.generate_locations import generate_locations


# ═════════════════════════════════════════════════════════════
# 1. CLASSIFICATION LOSS (Sigmoid Focal Loss)
# ═════════════════════════════════════════════════════════════
def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Sigmoid Focal Loss for multi-class classification without background.
    
    Args:
        logits: [N, C] raw logits from model
        targets: [N] class indices (-1 for negative samples)
        alpha: weighting factor
        gamma: focusing parameter
    
    Returns:
        loss (scalar)
    """
    N, C = logits.shape

    # Create one-hot encoding
    # positive samples (targets >= 0) get 1 at their class
    # negative samples (targets < 0) get 0 everywhere
    targets_onehot = torch.zeros_like(logits)
    pos_mask = targets >= 0
    targets_onehot[pos_mask, targets[pos_mask].long()] = 1.0

    # Compute probabilities
    prob = torch.sigmoid(logits)
    
    # Binary cross entropy
    ce = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction='none')

    # Focal weight
    p_t = prob * targets_onehot + (1 - prob) * (1 - targets_onehot)
    focal_weight = (1 - p_t) ** gamma
    
    loss = ce * focal_weight

    # Alpha weighting
    alpha_t = alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot)
    loss = alpha_t * loss

    return loss.sum()


# ═════════════════════════════════════════════════════════════
# 2. BOX REGRESSION LOSS (Smooth L1)
# ═════════════════════════════════════════════════════════════
def smooth_l1_loss(pred, target, beta=1.0):
    """
    Smooth L1 loss for bounding box regression.
    
    Args:
        pred: [N, 4] predicted offsets [l, t, r, b]
        target: [N, 4] target offsets [l, t, r, b]
        beta: smoothing parameter
    
    Returns:
        loss (scalar)
    """
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    )
    return loss.sum()


# ═════════════════════════════════════════════════════════════
# 3. MASK LOSS (Binary Cross Entropy)
# ═════════════════════════════════════════════════════════════
def mask_bce_loss(mask_logits, target_masks, box_mask):
    """
    Mask loss: BCE only on box region (ignore outside).
    
    Args:
        mask_logits: [N, H, W] logits from model
        target_masks: [N, H, W] target binary masks
        box_mask: [N, H, W] bool mask (True inside box, False outside)
    
    Returns:
        loss (scalar, averaged per positive sample)
    """
    # Compute BCE for all pixels
    bce = F.binary_cross_entropy_with_logits(
        mask_logits,
        target_masks,
        reduction='none'
    )
    
    # Only sum inside box region
    loss_per_sample = (bce * box_mask.float()).sum(dim=(1, 2)) / (
        box_mask.float().sum(dim=(1, 2)) + 1e-6
    )
    
    return loss_per_sample.sum()


# ═════════════════════════════════════════════════════════════
# MAIN LOSS CLASS
# ═════════════════════════════════════════════════════════════
class Model_Loss(nn.Module):
    """
    Combined loss for MVP-Seg:
    - Classification loss (focal)
    - Box regression loss (smooth L1)
    - Mask loss (BCE)
    """
    
    def __init__(
        self,
        num_classes=20,
        alpha_cls=1.0,
        alpha_box=1.0,
        alpha_mask=2.0,
        strides=[8, 16, 32],
        img_size=550,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_cls = alpha_cls
        self.alpha_box = alpha_box
        self.alpha_mask = alpha_mask
        self.strides = strides
        self.img_size = float(img_size)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with keys ['cls', 'box', 'coef', 'proto']
                cls: list of [B, C, Hi, Wi]
                box: list of [B, 4, Hi, Wi]
                coef: list of [B, P, Hi, Wi]
                proto: [B, P, H2/2, W2/2]
            
            targets: list of dicts with keys ['boxes', 'labels', 'masks']
                boxes: [N, 4] in [x1, y1, x2, y2] format
                labels: [N] in 0-19 range
                masks: [N, H, W] binary masks
        
        Returns:
            dict with keys ['loss', 'loss_cls', 'loss_box', 'loss_mask']
        """
        
        # ═════════════════════════════════════════
        # STEP 1: Flatten predictions across feature levels
        # ═════════════════════════════════════════
        cls_preds = flatten_predictions(outputs["cls"])    # [B, N, C]
        box_preds = flatten_predictions(outputs["box"])    # [B, N, 4]
        coef_preds = flatten_predictions(outputs["coef"])  # [B, N, P]
        proto = outputs["proto"]                           # [B, P, Hp, Wp]
        
        B, N, C = cls_preds.shape
        total_predictions = B * N
        
        # ═════════════════════════════════════════
        # STEP 2: Generate anchor locations
        # ══════════════════════════════��══════════
        locations = generate_locations(outputs["cls"], self.strides)
        locations = locations.to(cls_preds.device)
        
        # ═════════════════════════════════════════
        # STEP 3: Initialize accumulators
        # ═════════════════════════════════════════
        all_cls_loss = 0.0
        all_box_loss = 0.0
        all_mask_loss = 0.0
        total_num_pos = 0
        
        # ═════════════════════════════════════════
        # STEP 4: Process each image in batch
        # ═════════════════════════════════════════
        for i in range(B):
            gt_boxes = targets[i]["boxes"].to(cls_preds.device)    # [M, 4]
            gt_labels = targets[i]["labels"].to(cls_preds.device)  # [M]
            gt_masks = targets[i]["masks"].to(cls_preds.device)    # [M, H, W]
            
            # Skip if no objects
            if gt_boxes.shape[0] == 0:
                # Assign all locations as negative
                gt_cls_target = torch.full(
                    (N,),
                    -1,
                    dtype=torch.long,
                    device=cls_preds.device
                )
                all_cls_loss += sigmoid_focal_loss(cls_preds[i], gt_cls_target)
                continue
            
            # ─────────────────────────────────────
            # 4.1: Match locations to GT boxes
            # ─────────────────────────────────────
            matched_idx, pos_mask = match_locations(locations, gt_boxes)
            n_pos = pos_mask.sum().item()
            
            # Limit positive samples (optional, prevent too many)
            max_pos = 100
            if n_pos > max_pos:
                pos_idx = torch.where(pos_mask)[0]
                perm = torch.randperm(n_pos, device=pos_idx.device)[:max_pos]
                selected = pos_idx[perm]
                new_pos_mask = torch.zeros_like(pos_mask)
                new_pos_mask[selected] = True
                pos_mask = new_pos_mask
                n_pos = max_pos
            
            total_num_pos += n_pos
            
            # ─────────────────────────────────────
            # 4.2: CLASSIFICATION LOSS
            # ─────────────────────────────────────
            gt_cls_target = torch.full(
                (N,),
                -1,
                dtype=torch.long,
                device=cls_preds.device
            )
            if n_pos > 0:
                # Assign matched labels to positive locations
                gt_cls_target[pos_mask] = gt_labels[matched_idx[pos_mask]]
            
            all_cls_loss += sigmoid_focal_loss(cls_preds[i], gt_cls_target)
            
            # Skip rest if no positives
            if n_pos == 0:
                continue
            
            # ─────────────────────────────────────
            # 4.3: BOX REGRESSION LOSS
            # ─────────────────────────────────────
            pos_box_preds = box_preds[i][pos_mask]        # [n_pos, 4]
            pos_locs = locations[pos_mask]                # [n_pos, 2]
            matched_gt_boxes = gt_boxes[matched_idx[pos_mask]]  # [n_pos, 4]
            
            # Compute target offsets (in pixel space)
            l = pos_locs[:, 0] - matched_gt_boxes[:, 0]
            t = pos_locs[:, 1] - matched_gt_boxes[:, 1]
            r = matched_gt_boxes[:, 2] - pos_locs[:, 0]
            b = matched_gt_boxes[:, 3] - pos_locs[:, 1]
            
            target_ltrb = torch.stack([l, t, r, b], dim=1)
            
            # Clamp to avoid invalid boxes
            target_ltrb = target_ltrb.clamp(min=0.1)
            
            # ✅ NORMALIZE to 0-1 range (match model output scale)
            target_ltrb_norm = target_ltrb / self.img_size
            
            # Compute smooth L1 loss
            all_box_loss += smooth_l1_loss(pos_box_preds, target_ltrb_norm)
            
            # ─────────────────────────────────────
            # 4.4: MASK LOSS
            # ─────────────────────────────────────
            pos_coefs = coef_preds[i][pos_mask]  # [n_pos, P]
            P, p_h, p_w = proto.shape[1:]        # P, H/4, W/4
            
            # Generate predicted masks from coefficients
            # mask = coefficients @ prototype
            mask_logits = pos_coefs @ proto[i].view(P, -1)  # [n_pos, p_h*p_w]
            mask_logits = mask_logits.view(-1, p_h, p_w)    # [n_pos, p_h, p_w]
            
            # Resize GT masks to prototype size
            target_masks = F.interpolate(
                gt_masks[matched_idx[pos_mask]].unsqueeze(1).float(),
                size=(p_h, p_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [n_pos, p_h, p_w]
            
            # Create box mask (only compute loss inside box)
            H_img, W_img = gt_masks.shape[-2:]
            scale_x = p_w / W_img
            scale_y = p_h / H_img
            
            scaled_boxes = matched_gt_boxes.clone()
            scaled_boxes[:, [0, 2]] *= scale_x
            scaled_boxes[:, [1, 3]] *= scale_y
            
            # Create coordinate grids
            y_coords, x_coords = torch.meshgrid(
                torch.arange(p_h, device=mask_logits.device),
                torch.arange(p_w, device=mask_logits.device),
                indexing='ij'
            )
            
            y_coords = y_coords.unsqueeze(0)  # [1, p_h, p_w]
            x_coords = x_coords.unsqueeze(0)  # [1, p_h, p_w]
            
            # Create boolean mask for inside box
            box_mask = (
                (x_coords >= scaled_boxes[:, 0].view(-1, 1, 1)) &
                (x_coords <= scaled_boxes[:, 2].view(-1, 1, 1)) &
                (y_coords >= scaled_boxes[:, 1].view(-1, 1, 1)) &
                (y_coords <= scaled_boxes[:, 3].view(-1, 1, 1))
            )  # [n_pos, p_h, p_w]
            
            # Compute mask loss
            all_mask_loss += mask_bce_loss(mask_logits, target_masks, box_mask)
        
        # ═════════════════════════════════════════
        # STEP 5: Normalize and combine losses
        # ═════════════════════════════════════════
        divisor_cls = total_predictions
        divisor_box_mask = max(total_num_pos, 1)
        
        final_loss_cls = (all_cls_loss / divisor_cls) * self.alpha_cls
        final_loss_box = (all_box_loss / divisor_box_mask) * self.alpha_box
        final_loss_mask = (all_mask_loss / divisor_box_mask) * self.alpha_mask
        
        total_loss = final_loss_cls + final_loss_box + final_loss_mask
        
        return {
            "loss": total_loss,
            "loss_cls": final_loss_cls,
            "loss_box": final_loss_box,
            "loss_mask": final_loss_mask,
        }