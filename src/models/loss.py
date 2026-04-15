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
        logits  : [N, C] raw logits from model
        targets : [N] class indices (-1 for negative samples)
        alpha   : weighting factor
        gamma   : focusing parameter

    Returns:
        loss (scalar)
    """
    logits = logits.float()
    N, C = logits.shape

    # One-hot: positive locations get 1 at their class, negatives get 0 everywhere
    targets_onehot = torch.zeros_like(logits)
    pos_mask = targets >= 0
    if pos_mask.any():
        targets_onehot[pos_mask, targets[pos_mask].long()] = 1.0

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
        pred   : [N, 4] predicted offsets [l, t, r, b]
        target : [N, 4] target offsets [l, t, r, b]
        beta   : smoothing parameter

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
        mask_logits  : [N, H, W] logits from model
        target_masks : [N, H, W] target binary masks
        box_mask     : [N, H, W] bool mask (True inside box)

    Returns:
        loss (scalar, averaged per positive sample)
    """
    bce = F.binary_cross_entropy_with_logits(
        mask_logits,
        target_masks,
        reduction='none'
    )

    loss_per_sample = (bce * box_mask.float()).sum(dim=(1, 2)) / (
        box_mask.float().sum(dim=(1, 2)) + 1e-6
    )

    return loss_per_sample.sum()

# ═════════════════════════════════════════════════════════════
# HELPER: tính stride của từng location
# ═════════════════════════════════════════════════════════════
def _build_stride_tensor(level_sizes, strides, device):
    """
    Trả về tensor [N] chứa stride tương ứng với mỗi location.
    Dùng để normalize target ltrb: target_norm = target_pixel / stride
    → cả pred (softplus output ~0-10) và target_norm đều cùng scale.
    """
    parts = []
    for size, stride in zip(level_sizes, strides):
        parts.append(torch.full((int(size),), float(stride), device=device))
    return torch.cat(parts)  # [N]

# ═════════════════════════════════════════════════════════════
# MAIN LOSS CLASS
# ═════════════════════════════════════════════════════════════
class Model_Loss(nn.Module):
    """
    Combined loss for MVP-Seg:
      - Classification loss (Sigmoid Focal Loss)
      - Box regression loss (Smooth L1)
      - Mask loss (BCE inside box region)

    Weights theo YOLACT paper: alpha=1.0, beta=1.5, gamma=6.125
    """

    def __init__(
        self,
        num_classes=20,
        alpha_cls=1.0,
        alpha_box=1.5,    # tăng từ 1.5 để box loss đủ mạnh
        alpha_mask=6.125,
        strides=[8, 16, 32],
        img_size=550,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_cls   = alpha_cls
        self.alpha_box   = alpha_box
        self.alpha_mask  = alpha_mask
        self.strides     = strides
        self.img_size    = float(img_size)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict với keys ['cls', 'box', 'coef', 'proto']
                cls  : list of [B, C, Hi, Wi]
                box  : list of [B, 4, Hi, Wi]
                coef : list of [B, P, Hi, Wi]
                proto: [B, P, Hp, Wp]

            targets: list of dicts với keys ['boxes', 'labels', 'masks']
                boxes : [M, 4] in [x1, y1, x2, y2] format
                labels: [M] in 0-based class index
                masks : [M, H, W] binary masks

        Returns:
            dict với keys ['loss', 'loss_cls', 'loss_box', 'loss_mask']
        """

        # ═══════════════════════════════════════════════
        # STEP 1: Flatten predictions
        # ═══════════════════════════════════════════════
        cls_preds  = flatten_predictions(outputs["cls"])    # [B, N, C]
        box_preds  = flatten_predictions(outputs["box"])    # [B, N, 4]
        coef_preds = flatten_predictions(outputs["coef"])   # [B, N, P]
        proto      = outputs["proto"]                       # [B, P, Hp, Wp]
        level_sizes = [x.shape[2] * x.shape[3] for x in outputs["cls"]]

        B, N, C = cls_preds.shape

        # ═══════════════════════════════════════════════
        # STEP 2: Generate anchor locations
        # ═══════════════════════════════════════════════
        locations = generate_locations(outputs["cls"], self.strides)
        locations = locations.to(cls_preds.device)
        
        # [N] — stride tương ứng với từng location, dùng để normalize target
        stride_per_loc = _build_stride_tensor(level_sizes, self.strides, cls_preds.device)
        
        # ═══════════════════════════════════════════════
        # STEP 3: Khởi tạo accumulators
        # ═══════════════════════════════════════════════
        all_cls_loss          = 0.0
        all_box_loss          = 0.0
        all_mask_loss         = 0.0
        total_num_pos         = 0
        total_num_loc         = 0  # tổng locations dùng cho normalize cls

        # ═══════════════════════════════════════════════
        # STEP 4: Xử lý từng ảnh trong batch
        # ═══════════════════════════════════════════════
        for i in range(B):
            gt_boxes  = targets[i]["boxes"].to(cls_preds.device)   # [M, 4]
            gt_labels = targets[i]["labels"].to(cls_preds.device)  # [M]
            gt_masks  = targets[i]["masks"].to(cls_preds.device)   # [M, H, W]

            # ─────────────────────────────────────────
            # 4.1: Match locations → positive/negative
            # ─────────────────────────────────────────
            if gt_boxes.shape[0] > 0:
                matched_idx, pos_mask = match_locations(
                    locations,
                    gt_boxes,
                    strides=self.strides,
                    level_sizes=level_sizes,
                )
                n_pos = pos_mask.sum().item()
                total_num_pos += n_pos
            else:
                # Ảnh không có object: toàn bộ là negative
                pos_mask    = torch.zeros(N, dtype=torch.bool, device=cls_preds.device)
                matched_idx = torch.full((N,), -1, dtype=torch.long, device=cls_preds.device)
                n_pos       = 0

            # ─────────────────────────────────────────
            # 4.2: Classification loss
            # ─────────────────────────────────────────
            # Gán nhãn: positive → class index, negative → -1
            gt_cls_target = torch.full((N,), -1, dtype=torch.long, device=cls_preds.device)
            if n_pos > 0:
                gt_cls_target[pos_mask] = gt_labels[matched_idx[pos_mask]]

            # Negative mining: pos:neg = 1:3
            # Áp dụng nhất quán cho cả ảnh có và không có object
            # neg_mask = ~pos_mask
            # num_pos  = pos_mask.sum()
            # num_neg  = neg_mask.sum()
            # max_neg  = num_pos * 3 if num_pos > 0 else 100

            # if num_neg > max_neg:
            #     neg_idx      = torch.where(neg_mask)[0]
            #     perm         = torch.randperm(num_neg, device=neg_idx.device)[:max_neg]
            #     new_neg_mask = torch.zeros_like(neg_mask)
            #     new_neg_mask[neg_idx[perm]] = True
            #     neg_mask     = new_neg_mask

            # # Chỉ tính loss trên pos + selected neg
            # final_mask       = pos_mask | neg_mask
            # cls_pred_sampled = cls_preds[i][final_mask]     # [K, C]
            # gt_cls_sampled   = gt_cls_target[final_mask]    # [K]

            # all_cls_loss          += sigmoid_focal_loss(cls_pred_sampled, gt_cls_sampled)
            # total_num_cls_samples += final_mask.sum().item()
            # Áp dụng focal loss trên toàn bộ locations, nhưng chỉ có positive mới có target class, negative sẽ có target -1 và được ignore trong loss
            all_cls_loss += sigmoid_focal_loss(cls_preds[i], gt_cls_target)
            total_num_loc += N

            # Không có positive → bỏ qua box và mask loss
            if n_pos == 0:
                continue

            # ─────────────────────────────────────────
            # 4.3: Box regression loss
            # ─────────────────────────────────────────
            pos_box_preds    = box_preds[i][pos_mask]                  # [n_pos, 4]
            pos_locs         = locations[pos_mask]    
            pos_strides      = stride_per_loc[pos_mask]         # [n_pos]                 # [n_pos, 2]
            matched_gt_boxes = gt_boxes[matched_idx[pos_mask]]         # [n_pos, 4]

            # Target offset ltrb (pixel space)
            l = pos_locs[:, 0] - matched_gt_boxes[:, 0]
            t = pos_locs[:, 1] - matched_gt_boxes[:, 1]
            r = matched_gt_boxes[:, 2] - pos_locs[:, 0]
            b = matched_gt_boxes[:, 3] - pos_locs[:, 1]

            target_ltrb_px = torch.stack([l, t, r, b], dim=1).clamp(min=0.0)

            # Normalize về [0,1] để khớp với scale của box_preds
            target_ltrb_norm = target_ltrb_px / pos_strides.unsqueeze(1)
            # SỬA: KHÔNG chia cho self.img_size nữa vì box_preds dùng softplus
            # SỬA: Giảm beta của smooth_l1_loss xuống 0.1 để phù hợp với pixel scale
            all_box_loss += smooth_l1_loss(pos_box_preds, target_ltrb_norm, beta=0.1)

            # ─────────────────────────────────────────
            # 4.4: Mask loss
            # ─────────────────────────────────────────
            pos_coefs = coef_preds[i][pos_mask]   # [n_pos, P]
            P, p_h, p_w = proto.shape[1:]

            # M = sigmoid(coef @ proto^T) — theo proposal trang 8
            mask_logits = pos_coefs @ proto[i].view(P, -1)  # [n_pos, p_h*p_w]
            mask_logits = mask_logits.view(-1, p_h, p_w)    # [n_pos, p_h, p_w]

            # Resize GT masks xuống prototype size
            target_masks = F.interpolate(
                gt_masks[matched_idx[pos_mask]].unsqueeze(1).float(),
                size=(p_h, p_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [n_pos, p_h, p_w]

            # Scale box về prototype space để crop mask
            H_img, W_img = gt_masks.shape[-2:]
            scale_x = p_w / W_img
            scale_y = p_h / H_img

            scaled_boxes = matched_gt_boxes.clone()
            scaled_boxes[:, [0, 2]] *= scale_x
            scaled_boxes[:, [1, 3]] *= scale_y

            # Coordinate grid
            y_coords, x_coords = torch.meshgrid(
                torch.arange(p_h, device=mask_logits.device),
                torch.arange(p_w, device=mask_logits.device),
                indexing='ij'
            )
            y_coords = y_coords.unsqueeze(0)  # [1, p_h, p_w]
            x_coords = x_coords.unsqueeze(0)  # [1, p_h, p_w]

            # Boolean mask: True trong vùng box
            box_mask = (
                (x_coords >= scaled_boxes[:, 0].view(-1, 1, 1)) &
                (x_coords <= scaled_boxes[:, 2].view(-1, 1, 1)) &
                (y_coords >= scaled_boxes[:, 1].view(-1, 1, 1)) &
                (y_coords <= scaled_boxes[:, 3].view(-1, 1, 1))
            )  # [n_pos, p_h, p_w]

            all_mask_loss += mask_bce_loss(mask_logits, target_masks, box_mask)

        # ═══════════════════════════════════════════════
        # STEP 5: Normalize và combine losses
        # ═══════════════════════════════════════════════
        # cls: chia theo số location thực tế được sample (nhất quán với negative mining)
        # divisor_cls      = max(total_num_cls_samples, 1)
        # divisor_box_mask = max(total_num_pos, 1)
        # SỬA: Tất cả đều normalize dựa trên số lượng POSITIVE samples
        divisor_cls  = max(total_num_loc, 1)
        divisor_pos  = max(total_num_pos, 1)

        final_loss_cls  = (all_cls_loss  / divisor_cls) * self.alpha_cls
        final_loss_box  = (all_box_loss  / divisor_pos) * self.alpha_box
        final_loss_mask = (all_mask_loss / divisor_pos) * self.alpha_mask

        total_loss = final_loss_cls + final_loss_box + final_loss_mask

        return {
            "loss":      total_loss,
            "loss_cls":  final_loss_cls,
            "loss_box":  final_loss_box,
            "loss_mask": final_loss_mask,
        }
