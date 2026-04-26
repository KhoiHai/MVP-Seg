"""
sbd_eval.py — Evaluation script cho MVP-Seg trên SBD dataset
=============================================================
Dựa theo Proposal MVP-Seg (22/3/2026), Chương 5.1.4 — Đánh giá hiệu năng:

  Metrics bắt buộc (Bảng 4 trong proposal):
    - mAP       : AP trung bình trên IoU 0.5 → 0.95, bước 0.05  
    - AP50      : AP tại IoU = 0.5
    - AP75      : AP tại IoU = 0.75
    - APS       : AP với object nhỏ  (area < 32²)
    - APM       : AP với object trung bình (32² ≤ area < 96²)
    - APL       : AP với object lớn  (area ≥ 96²)
    - FPS       : Số khung hình/giây trên val set

  Proposal KHÔNG đề cập mIoU mask — không đưa vào làm metric chính.

  Classification loss trong proposal dùng Softmax Cross-Entropy (trang 8),
  nhưng code loss.py đang dùng Sigmoid Focal Loss → decode dùng sigmoid,
  không phải softmax (bám theo code thực tế).

  Dataset train thực tế: SBD (20 class) — đây là dataset test script này.
  Dataset theo proposal (COCO 80 class, Cityscapes) dùng cho giai đoạn sau.

Cách dùng trên Colab:
  !python sbd_eval.py
  hoặc:
  from sbd_eval import evaluate_sbd
  results = evaluate_sbd(model_path="checkpoints/best.pth", data_root="/content/SBD")
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms, box_iou

from src.models.mvp_seg import MVP_Seg
from src.dataset.sbd_dataset import get_sbd_dataloaders
from src.utils.flatten_predictions import flatten_predictions
from src.utils.generate_locations import generate_locations


# ════════════════════════════════════════════════════════════════
# HẰNG SỐ
# ════════════════════════════════════════════════════════════════

# 20 class của SBD (Pascal VOC, index 0–19)
SBD_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird",   "boat",        "bottle",
    "bus",       "car",     "cat",    "chair",        "cow",
    "diningtable","dog",    "horse",  "motorbike",    "person",
    "pottedplant","sheep",  "sofa",   "train",         "tvmonitor",
]

# Ngưỡng IoU cho COCO-style AP: 0.50, 0.55, ..., 0.95
COCO_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

# Ngưỡng diện tích cho APS / APM / APL (theo chuẩn COCO)
AREA_SMALL  = 32 ** 2       # < 1024 px²
AREA_MEDIUM = 96 ** 2       # < 9216 px²
# Lớn: >= 9216 px²

# Theo proposal (Bảng 1): Top-k = 200, NMS thresh = 0.5
TOP_K       = 200
NMS_THRESH  = 0.5

# Strides tương ứng với N2, N3, N4 (khớp với train.py / loss.py)
STRIDES     = [8, 16, 32]

# Kích thước ảnh input (theo proposal và config train)
IMG_SIZE    = 550


# ════════════════════════════════════════════════════════════════
# PHẦN 1: DECODE PREDICTIONS
# ════════════════════════════════════════════════════════════════

def decode_predictions(outputs, score_thresh=0.05):
    """
    Giải mã output của MVP_Seg.forward() thành danh sách box + mask.

    Theo proposal (trang 7-8):
      1. Flatten tất cả prediction từ N2, N3, N4
      2. Chọn Top-k (200) prediction có score cao nhất
      3. Áp dụng Fast NMS loại bỏ duplicate
      4. Tái tạo mask: M = sigmoid(coef @ proto^T)

    Args:
        outputs      (dict): kết quả model.forward(), gồm:
                             'cls', 'box', 'coef' → list of [B, C/4/P, Hi, Wi]
                             'proto'              → [B, P, Hp, Wp]
        score_thresh (float): Ngưỡng lọc sơ bộ trước Top-k
                              (đặt thấp = 0.05 để AP tính đủ recall)

    Returns:
        list[dict]: mỗi dict gồm:
            'boxes'  : [K, 4] xyxy pixel space
            'scores' : [K]    confidence score (sigmoid max)
            'labels' : [K]    class index 0-based
            'masks'  : [K, Hp, Wp] soft mask [0,1]
            'areas'  : [K]    diện tích box (pixel²) để phân loại S/M/L
    """
    # Flatten multi-scale predictions
    cls_preds  = flatten_predictions(outputs["cls"])    # [B, N, C]
    box_preds  = flatten_predictions(outputs["box"])    # [B, N, 4]  — ltrb normalize
    coef_preds = flatten_predictions(outputs["coef"])   # [B, N, P]
    proto      = outputs["proto"]                       # [B, P, Hp, Wp]

    # Sinh location grid tương ứng với từng feature level
    # generate_locations trả về [N_total, 2] — tọa độ (px, py) pixel space
    locations = generate_locations(outputs["cls"], STRIDES)  # [N, 2]
    locations = locations.to(cls_preds.device)

    level_sizes = [x.shape[2] * x.shape[3] for x in outputs["cls"]]
    stride_tensor = []
    for size, s_val in zip(level_sizes, STRIDES):
        stride_tensor.append(torch.full((int(size),), float(s_val), device=cls_preds.device))
    stride_tensor = torch.cat(stride_tensor)

    B = cls_preds.shape[0]
    results = []

    for i in range(B):
        # ── Score: sigmoid vì code dùng Sigmoid Focal Loss ────────
        scores_all = torch.sigmoid(cls_preds[i])        # [N, C]
        scores, labels = scores_all.max(dim=-1)         # [N], [N]

        # Lọc sơ bộ bằng score_thresh để giảm tải
        keep = scores > score_thresh
        if keep.sum() == 0:
            results.append({
                "boxes":  torch.zeros((0, 4), device=cls_preds.device),
                "scores": torch.zeros((0,),   device=cls_preds.device),
                "labels": torch.zeros((0,), dtype=torch.long, device=cls_preds.device),
                "masks":  torch.zeros((0, proto.shape[2], proto.shape[3]),
                                      device=cls_preds.device),
                "areas":  torch.zeros((0,), device=cls_preds.device),
            })
            continue

        scores_f = scores[keep]
        labels_f = labels[keep]
        # boxes_f  = box_preds[i][keep]      # [K, 4] ltrb normalize
        coefs_f  = coef_preds[i][keep]     # [K, P]
        locs_f   = locations[keep]         # [K, 2] (px, py)
        pos_strides = stride_tensor[keep].unsqueeze(1)
        boxes_f  = box_preds[i][keep] * pos_strides  # [K, 4] ltrb normalize → scale theo stride để thành pixel space

        # ── Top-k (theo proposal: k = 200) ───────────────────────
        k = min(TOP_K, scores_f.shape[0])
        topk_scores, topk_idx = scores_f.topk(k)

        boxes_f  = boxes_f[topk_idx]
        labels_f = labels_f[topk_idx]
        coefs_f  = coefs_f[topk_idx]
        locs_f   = locs_f[topk_idx]

        # ── Chuyển box từ ltrb normalize sang xyxy pixel ──────────
        # box_preds là [l, t, r, b] đã normalize chia cho img_size
        # location (px, py) là tâm của vùng dự đoán
        # x1 = px - l*IMG_SIZE,  y1 = py - t*IMG_SIZE, ...
        l = boxes_f[:, 0]
        t = boxes_f[:, 1]
        r = boxes_f[:, 2]
        b = boxes_f[:, 3]

        px = locs_f[:, 0].float()
        py = locs_f[:, 1].float()

        x1 = (px - l).clamp(0, IMG_SIZE)
        y1 = (py - t).clamp(0, IMG_SIZE)
        x2 = (px + r).clamp(0, IMG_SIZE)
        y2 = (py + b).clamp(0, IMG_SIZE)

        xyxy_boxes = torch.stack([x1, y1, x2, y2], dim=1)  # [K, 4]

        # ── Fast NMS per-class (theo proposal trang 7) ───────────
        # Dùng per-class NMS để tránh suppress object khác class
        keep_final = []
        for cls_id in labels_f.unique():
            cls_mask   = labels_f == cls_id
            cls_boxes  = xyxy_boxes[cls_mask]
            cls_scores = topk_scores[cls_mask]
            keep_cls   = nms(cls_boxes, cls_scores, NMS_THRESH)
            global_idx = cls_mask.nonzero(as_tuple=True)[0]
            keep_final.append(global_idx[keep_cls])

        if len(keep_final) == 0:
            results.append({
                "boxes":  torch.zeros((0, 4), device=cls_preds.device),
                "scores": torch.zeros((0,),   device=cls_preds.device),
                "labels": torch.zeros((0,), dtype=torch.long, device=cls_preds.device),
                "masks":  torch.zeros((0, proto.shape[2], proto.shape[3]),
                                      device=cls_preds.device),
                "areas":  torch.zeros((0,), device=cls_preds.device),
            })
            continue

        keep_final   = torch.cat(keep_final)
        final_boxes  = xyxy_boxes[keep_final]
        final_labels = labels_f[keep_final]
        final_scores = topk_scores[keep_final]
        final_coefs  = coefs_f[keep_final]

        # ── Tái tạo mask: M = sigmoid(coef @ proto^T) ────────────
        # Công thức từ proposal trang 8: M = σ(C·P^T)
        P, Hp, Wp  = proto.shape[1:]
        proto_flat = proto[i].view(P, -1).T          # [Hp*Wp, P]
        mask_logits = proto_flat @ final_coefs.T      # [Hp*Wp, K]
        mask_logits = mask_logits.T.view(-1, Hp, Wp)  # [K, Hp, Wp]
        pred_masks  = torch.sigmoid(mask_logits)      # [K, Hp, Wp]

        # Diện tích box để phân loại S/M/L theo chuẩn COCO
        areas = (final_boxes[:, 2] - final_boxes[:, 0]) * \
                (final_boxes[:, 3] - final_boxes[:, 1])   # [K]

        results.append({
            "boxes":  final_boxes,
            "scores": final_scores,
            "labels": final_labels,
            "masks":  pred_masks,
            "areas":  areas,
        })

    return results


# ════════════════════════════════════════════════════════════════
# PHẦN 2: TÍNH AP THEO COCO-STYLE
# ════════════════════════════════════════════════════════════════

def compute_ap_at_iou(tp, fp, n_gt):
    """
    Tính AP tại 1 ngưỡng IoU bằng 101-point interpolation.

    Args:
        tp   (np.ndarray): array TP đã sort theo score giảm dần
        fp   (np.ndarray): array FP đã sort theo score giảm dần
        n_gt (int)       : tổng số ground truth object

    Returns:
        ap (float)
    """
    if n_gt == 0:
        return float("nan")   # Không tính AP nếu class không có GT

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall    = cum_tp / (n_gt + 1e-10)
    precision = cum_tp / (cum_tp + cum_fp + 1e-10)

    # 101-point interpolation (VOC2010+ / COCO standard)
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        prec = precision[recall >= thr]
        ap  += prec.max() if len(prec) > 0 else 0.0
    return ap / 101.0


def match_preds_to_gt_at_iou(
    pred_boxes,
    pred_labels,
    pred_scores,
    gt_boxes,
    gt_labels,
    iou_thresh,
):
    """
    Greedy matching: sort prediction theo score giảm dần,
    mỗi GT chỉ được match 1 lần, cùng class.

    Returns:
        tp     (np.ndarray [K]) — sắp xếp theo thứ tự score giảm dần
        fp     (np.ndarray [K])
        scores (np.ndarray [K])
    """
    K = len(pred_boxes)
    M = len(gt_boxes)

    tp     = np.zeros(K)
    fp     = np.zeros(K)

    if M == 0:
        fp[:] = 1
        order = pred_scores.argsort(descending=True).cpu().numpy()
        return tp, fp, pred_scores[order].cpu().numpy()

    order      = pred_scores.argsort(descending=True)
    iou_matrix = box_iou(pred_boxes, gt_boxes)   # [K, M]
    matched_gt = set()

    for rank, pidx in enumerate(order):
        cls        = pred_labels[pidx].item()
        gt_cls_idx = (gt_labels == cls).nonzero(as_tuple=True)[0]

        if len(gt_cls_idx) == 0:
            fp[rank] = 1
            continue

        ious     = iou_matrix[pidx, gt_cls_idx]
        best_iou, best_local = ious.max(0)
        best_gt  = gt_cls_idx[best_local.item()].item()

        if best_iou.item() >= iou_thresh and best_gt not in matched_gt:
            tp[rank] = 1
            matched_gt.add(best_gt)
        else:
            fp[rank] = 1

    scores_sorted = pred_scores[order].cpu().numpy()
    return tp, fp, scores_sorted


# ════════════════════════════════════════════════════════════════
# PHẦN 3: VÒNG LẶP EVAL CHÍNH
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_sbd(
    model_path,
    data_root    = "/content/SBD",
    num_classes  = 20,
    num_prototypes = 32,
    score_thresh = 0.05,
    device       = None,
    verbose      = True,
):
    """
    Đánh giá MVP-Seg trên SBD val set.
    Metrics theo Bảng 4 của proposal (trang 10):
      mAP (IoU 0.5:0.95), AP50, AP75, APS, APM, APL, FPS.

    Args:
        model_path   (str)  : Đường dẫn checkpoint .pth
        data_root    (str)  : Thư mục SBD (phải khớp config train)
        num_classes  (int)  : 20 cho SBD
        score_thresh (float): Ngưỡng lọc sơ bộ (thấp để tính đủ recall)
        device       (str)  : 'cuda' / 'cpu' (auto nếu None)
        verbose      (bool) : In log

    Returns:
        dict gồm: mAP, AP50, AP75, APS, APM, APL, fps, AP_per_class
    """
    # ── Device ───────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"[INFO] Device: {device}")

    # ── Load model ────────────────────────────────────────────────
    model = MVP_Seg(
        model_name     = "nvidia/MambaVision-T-1K",
        pretrained     = False,    # Không load pretrained khi eval
        num_classes    = num_classes,
        num_prototypes = num_prototypes,       # Theo Bảng 1 proposal: P = 32
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    # Hỗ trợ cả lưu dict (train.py) và lưu state_dict thuần
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        if verbose:
            print(f"[INFO] Loaded checkpoint epoch {ckpt.get('epoch','?')}: {model_path}")
    else:
        model.load_state_dict(ckpt)
        if verbose:
            print(f"[INFO] Loaded state_dict: {model_path}")

    model.eval()

    # ── Dataloader ────────────────────────────────────────────────
    # batch_size=1 để eval chính xác từng ảnh và đo FPS đơn lẻ
    _, val_loader = get_sbd_dataloaders(
        root        = data_root,
        batch_size  = 1,
        num_workers = 2,
        img_size    = IMG_SIZE,
        verbose     = verbose,
    )

    # ── Accumulator ───────────────────────────────────────────────
    # acc_tp/fp/scores[iou_idx][class] = list các giá trị
    n_iou = len(COCO_IOU_THRESHOLDS)

    acc_tp     = [[[] for _ in range(num_classes)] for _ in range(n_iou)]
    acc_fp     = [[[] for _ in range(num_classes)] for _ in range(n_iou)]
    acc_scores = [[[] for _ in range(num_classes)] for _ in range(n_iou)]

    n_gt_per_class = np.zeros(num_classes, dtype=int)

    # APS/APM/APL tại IoU=0.5 riêng
    acc_tp_s = [[] for _ in range(num_classes)]
    acc_fp_s = [[] for _ in range(num_classes)]
    acc_tp_m = [[] for _ in range(num_classes)]
    acc_fp_m = [[] for _ in range(num_classes)]
    acc_tp_l = [[] for _ in range(num_classes)]
    acc_fp_l = [[] for _ in range(num_classes)]
    n_gt_s   = np.zeros(num_classes, dtype=int)
    n_gt_m   = np.zeros(num_classes, dtype=int)
    n_gt_l   = np.zeros(num_classes, dtype=int)

    fps_list = []
    n_images = 0

    if verbose:
        print(f"\n[INFO] Evaluating {len(val_loader)} val images...\n")

    # ── Inference loop ────────────────────────────────────────────
    for batch_idx, batch in enumerate(val_loader):
        if batch is None:
            continue

        images, targets = batch
        images = images.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        outputs = model(images)

        if device == "cuda":
            torch.cuda.synchronize()
        fps_list.append(1.0 / (time.time() - t0 + 1e-9))

        detections = decode_predictions(outputs, score_thresh=score_thresh)

        for det, tgt in zip(detections, targets):
            n_images += 1

            gt_boxes  = tgt["boxes"].to(device)    # [M, 4] xyxy
            gt_labels = tgt["labels"].to(device)   # [M]

            # Đếm GT per class
            for c in gt_labels.cpu().numpy():
                n_gt_per_class[int(c)] += 1

            # Diện tích GT box để đếm S/M/L
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
                       (gt_boxes[:, 3] - gt_boxes[:, 1])

            for c, a in zip(gt_labels.cpu().numpy(), gt_areas.cpu().numpy()):
                c = int(c)
                if a < AREA_SMALL:
                    n_gt_s[c] += 1
                elif a < AREA_MEDIUM:
                    n_gt_m[c] += 1
                else:
                    n_gt_l[c] += 1

            pred_boxes  = det["boxes"]
            pred_labels = det["labels"]
            pred_scores = det["scores"]
            pred_areas  = det["areas"]

            if len(pred_boxes) == 0:
                continue

            # ── Match tại từng IoU threshold ─────────────────────
            for iou_idx, iou_thr in enumerate(COCO_IOU_THRESHOLDS):
                tp, fp, sc = match_preds_to_gt_at_iou(
                    pred_boxes, pred_labels, pred_scores,
                    gt_boxes, gt_labels,
                    iou_thresh=float(iou_thr),
                )
                # Phân bổ vào từng class theo thứ tự sort
                order = pred_scores.argsort(descending=True).cpu().numpy()
                for rank, pidx in enumerate(order):
                    c = int(pred_labels[pidx].item())
                    acc_tp[iou_idx][c].append(tp[rank])
                    acc_fp[iou_idx][c].append(fp[rank])
                    acc_scores[iou_idx][c].append(sc[rank])

            # ── APS/APM/APL tại IoU=0.5 ──────────────────────────
            for area_min, area_max, atp, afp in [
                (0,           AREA_SMALL,       acc_tp_s, acc_fp_s),
                (AREA_SMALL,  AREA_MEDIUM,      acc_tp_m, acc_fp_m),
                (AREA_MEDIUM, float("inf"),     acc_tp_l, acc_fp_l),
            ]:
                pred_sz = (pred_areas >= area_min) & (pred_areas < area_max)
                if pred_sz.sum() == 0:
                    continue
                gt_sz = (gt_areas >= area_min) & (gt_areas < area_max)
                gb = gt_boxes[gt_sz]  if gt_sz.sum() > 0 else gt_boxes[:0]
                gl = gt_labels[gt_sz] if gt_sz.sum() > 0 else gt_labels[:0]

                pb = pred_boxes[pred_sz]
                pl = pred_labels[pred_sz]
                ps = pred_scores[pred_sz]

                tp_sz, fp_sz, _ = match_preds_to_gt_at_iou(
                    pb, pl, ps, gb, gl, iou_thresh=0.5
                )
                order_sz = ps.argsort(descending=True).cpu().numpy()
                for rank, pidx in enumerate(order_sz):
                    c = int(pl[pidx].item())
                    atp[c].append(tp_sz[rank])
                    afp[c].append(fp_sz[rank])

        if verbose and (batch_idx + 1) % 200 == 0:
            print(f"  [{batch_idx + 1}/{len(val_loader)}] images processed...")

    # ════════════════════════════════════════════════════════════
    # PHẦN 4: TỔNG HỢP METRICS
    # ════════════════════════════════════════════════════════════

    # ap_matrix[iou_idx, class_idx] = AP tại IoU threshold đó
    ap_matrix = np.full((n_iou, num_classes), np.nan)

    for iou_idx in range(n_iou):
        for c in range(num_classes):
            if len(acc_tp[iou_idx][c]) == 0:
                continue
            scores_c = np.array(acc_scores[iou_idx][c])
            tp_c     = np.array(acc_tp[iou_idx][c])
            fp_c     = np.array(acc_fp[iou_idx][c])
            sort_idx = np.argsort(-scores_c)
            ap_matrix[iou_idx, c] = compute_ap_at_iou(
                tp_c[sort_idx], fp_c[sort_idx], n_gt_per_class[c]
            )

    # mAP = nanmean trên cả 10 IoU threshold và 20 class
    map_score = float(np.nanmean(ap_matrix))
    ap50      = float(np.nanmean(ap_matrix[0, :]))   # index 0 = IoU 0.50
    ap75      = float(np.nanmean(ap_matrix[5, :]))   # index 5 = IoU 0.75

    # AP per class (trung bình 10 IoU thresholds)
    ap_per_class = {
        SBD_CLASS_NAMES[c]: float(np.nanmean(ap_matrix[:, c]))
        for c in range(num_classes)
    }

    # APS, APM, APL
    def _size_ap(atp, afp, n_gt_sz):
        vals = []
        for c in range(num_classes):
            if n_gt_sz[c] == 0:
                continue
            if len(atp[c]) == 0:
                vals.append(0.0)
                continue
            tp_c = np.array(atp[c])
            fp_c = np.array(afp[c])
            vals.append(compute_ap_at_iou(tp_c, fp_c, n_gt_sz[c]))
        return float(np.mean(vals)) if vals else 0.0

    aps_score = _size_ap(acc_tp_s, acc_fp_s, n_gt_s)
    apm_score = _size_ap(acc_tp_m, acc_fp_m, n_gt_m)
    apl_score = _size_ap(acc_tp_l, acc_fp_l, n_gt_l)

    mean_fps = float(np.mean(fps_list)) if fps_list else 0.0

    # ── In kết quả ────────────────────────────────────────────────
    if verbose:
        sep = "═" * 55
        print(f"\n{sep}")
        print("  MVP-Seg — Evaluation Results on SBD Val Set")
        print(f"  (Metrics theo Bảng 4, Proposal trang 10)")
        print(sep)
        print(f"  {'Metric':<22} {'Value':>10}")
        print("─" * 55)
        print(f"  {'mAP (IoU 0.5:0.95)':<22} {map_score*100:>9.2f}%")
        print(f"  {'AP50':<22} {ap50*100:>9.2f}%")
        print(f"  {'AP75':<22} {ap75*100:>9.2f}%")
        print(f"  {'APS (small < 32²)':<22} {aps_score*100:>9.2f}%")
        print(f"  {'APM (32² – 96²)':<22} {apm_score*100:>9.2f}%")
        print(f"  {'APL (large ≥ 96²)':<22} {apl_score*100:>9.2f}%")
        print(f"  {'FPS':<22} {mean_fps:>9.1f}")
        print(f"  {'Total images':<22} {n_images:>10}")
        print(sep)

        print("\n  AP per Class (mean over IoU 0.5:0.95):")
        print("─" * 55)
        for cls_name, ap in ap_per_class.items():
            bar = "█" * int(ap * 25)
            print(f"  {cls_name:<15} {ap*100:>6.2f}%  {bar}")
        print(sep)

    return {
        "mAP":          map_score,
        "AP50":         ap50,
        "AP75":         ap75,
        "APS":          aps_score,
        "APM":          apm_score,
        "APL":          apl_score,
        "fps":          mean_fps,
        "AP_per_class": ap_per_class,
        "n_images":     n_images,
    }


# ════════════════════════════════════════════════════════════════
# PHẦN 5: VISUALIZATION (tùy chọn — hữu ích trên Colab/Kaggle)
# ════════════════════════════════════════════════════════════════

# Bảng màu 20 instance — mỗi màu là RGBA
_VIS_COLORS = [
    (1.00, 0.20, 0.20),  # red
    (0.20, 0.80, 0.20),  # green
    (0.20, 0.40, 1.00),  # blue
    (1.00, 0.80, 0.00),  # yellow
    (1.00, 0.20, 1.00),  # magenta
    (0.00, 0.90, 0.90),  # cyan
    (1.00, 0.50, 0.00),  # orange
    (0.60, 0.20, 1.00),  # purple
    (0.00, 0.60, 0.30),  # dark green
    (0.80, 0.10, 0.30),  # crimson
    (0.20, 0.60, 1.00),  # sky blue
    (0.70, 0.70, 0.00),  # olive
    (0.00, 0.70, 0.70),  # teal
    (0.80, 0.00, 0.60),  # violet
    (1.00, 0.60, 0.70),  # pink
    (0.50, 0.90, 0.50),  # light green
    (0.60, 0.70, 1.00),  # light blue
    (1.00, 0.90, 0.50),  # light yellow
    (0.50, 1.00, 0.90),  # light cyan
    (0.90, 0.60, 1.00),  # light purple
]


def visualize_predictions(
    model,
    val_loader,
    device,
    num_samples  = 20,
    score_thresh = 0.3,
    mask_thresh  = 0.5,
    save_dir     = None,   # None → plt.show(); str → lưu PNG vào thư mục đó
    zip_path     = None,   # str → sau khi lưu xong sẽ zip toàn bộ save_dir vào đây
                           # Ví dụ: "/kaggle/working/vis_results.zip"
                           # Chỉ có tác dụng khi save_dir != None
):
    """
    Vẽ kết quả phân đoạn MVP-Seg lên ảnh gốc — chỉ 1 panel prediction
    (không có panel ảnh gốc bên cạnh), giống style figure YOLACT paper.

    Mỗi instance được tô màu overlay bán trong suốt + đường viền contour
    + bounding box + label text, nhất quán với hình minh họa bài báo.

    Args:
        model        : MVP_Seg đã load weight, ở chế độ eval.
        val_loader   : DataLoader của SBD val set (batch_size=1).
        device       : 'cuda' / 'cpu'.
        num_samples  : Số ảnh muốn vẽ (mặc định 20).
        score_thresh : Ngưỡng confidence để hiển thị detection.
        mask_thresh  : Ngưỡng nhị phân hóa soft mask.
        save_dir     : Thư mục lưu PNG. None → hiển thị trực tiếp.
        zip_path     : Nếu đặt, zip toàn bộ save_dir vào file này sau khi vẽ xong.
    """
    import zipfile
    import matplotlib
    matplotlib.use("Agg")          # backend không cần GUI — an toàn trên Kaggle/Colab
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    model.eval()
    count = 0
    saved_files = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in val_loader:
            if batch is None or count >= num_samples:
                break

            images, targets = batch
            images = images.to(device)
            outputs = model(images)
            dets = decode_predictions(outputs, score_thresh=score_thresh)

            for img_t, det in zip(images, dets):
                if count >= num_samples:
                    break

                # ── Denormalize về [0,1] RGB ──────────────────────
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
                img_np = (img_t * std + mean).clamp(0, 1).permute(1,2,0).cpu().numpy()
                H, W   = img_np.shape[:2]

                # ── Tạo canvas: 1 panel duy nhất ──────────────────
                fig, ax = plt.subplots(1, 1, figsize=(7, 7))
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

                # Vẽ ảnh gốc làm nền
                ax.imshow(img_np, interpolation="bilinear")
                ax.set_xlim(0, W)
                ax.set_ylim(H, 0)
                ax.axis("off")

                n_det = len(det["boxes"])
                legend_patches = []

                for j in range(n_det):
                    color  = _VIS_COLORS[j % len(_VIS_COLORS)]
                    cls_id = det["labels"][j].item()
                    score  = det["scores"][j].item()

                    # ── Upsample mask lên kích thước ảnh ──────────
                    pm = det["masks"][j]          # [Hp, Wp]
                    pm = F.interpolate(
                        pm.unsqueeze(0).unsqueeze(0).to(device),
                        size=(H, W), mode="bilinear", align_corners=False
                    ).squeeze().cpu()

                    # Crop mask ngoài bounding box (proposal trang 8)
                    x1, y1, x2, y2 = det["boxes"][j].cpu().numpy()
                    mask_bin = (pm > mask_thresh).float()
                    x1i = max(0, int(x1));  y1i = max(0, int(y1))
                    x2i = min(W, int(x2));  y2i = min(H, int(y2))
                    cropped = torch.zeros_like(mask_bin)
                    cropped[y1i:y2i, x1i:x2i] = mask_bin[y1i:y2i, x1i:x2i]
                    mask_np = cropped.numpy().astype(bool)

                    # ── Overlay màu bán trong suốt ─────────────────
                    overlay = np.zeros((H, W, 4), dtype=np.float32)
                    overlay[mask_np] = [*color, 0.45]
                    ax.imshow(overlay, interpolation="nearest")

                    # ── Contour viền mask ──────────────────────────
                    ax.contour(
                        mask_np.astype(np.uint8),
                        levels=[0.5],
                        colors=[color],
                        linewidths=1.5,
                    )

                    # ── Bounding box ───────────────────────────────
                    rect = mpatches.FancyBboxPatch(
                        (x1, y1), x2 - x1, y2 - y1,
                        boxstyle="square,pad=0",
                        linewidth=1.8,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax.add_patch(rect)

                    # ── Label text ─────────────────────────────────
                    label_str = f"{SBD_CLASS_NAMES[cls_id]}: {score:.2f}"
                    ax.text(
                        x1 + 2, max(y1 - 3, 6),
                        label_str,
                        fontsize=7.5, color="white", fontweight="bold",
                        va="bottom",
                        bbox=dict(
                            facecolor=color, alpha=0.82,
                            pad=1.5, edgecolor="none",
                            boxstyle="round,pad=0.2"
                        ),
                    )

                    legend_patches.append(
                        mpatches.Patch(color=color, label=label_str)
                    )

                # ── Legend (tối đa 10 entries, 2 cột) ─────────────
                if legend_patches:
                    ax.legend(
                        handles=legend_patches[:10],
                        loc="upper right",
                        fontsize=6.5,
                        framealpha=0.65,
                        ncol=2,
                        handlelength=1.2,
                        borderpad=0.5,
                    )

                # ── Lưu hoặc hiển thị ──────────────────────────────
                if save_dir:
                    fpath = os.path.join(save_dir, f"pred_{count:04d}.png")
                    plt.savefig(fpath, dpi=150, bbox_inches="tight",
                                pad_inches=0, facecolor="black")
                    saved_files.append(fpath)
                    print(f"  [Saved] {fpath}  ({n_det} detections)")
                else:
                    plt.show()

                plt.close(fig)
                count += 1

    print(f"\n[VIS] Đã vẽ {count} ảnh.")

    # ── Zip toàn bộ thư mục kết quả ───────────────────────────────
    if save_dir and zip_path and saved_files:
        import zipfile
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath in saved_files:
                zf.write(fpath, arcname=os.path.basename(fpath))
        size_kb = os.path.getsize(zip_path) / 1024
        print(f"[VIS] Đã tạo zip: {zip_path}  ({size_kb:.1f} KB, {len(saved_files)} files)")


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

def eval(CONFIG):

    # CONFIG = {
    #     # Checkpoint từ train.py (lưu dạng {"model_state": ..., "epoch": ...})
    #     "model_path":    "checkpoints/best.pth",

    #     # Phải khớp với data_root trong config train
    #     "data_root":     "/content/SBD",

    #     # Số class SBD = 20
    #     "num_classes":   20,

    #     # Ngưỡng score sơ bộ khi decode
    #     # Đặt thấp (0.05) để giữ đủ recall khi tính AP
    #     "score_thresh":  0.05,

    #     # Visualization
    #     "visualize":     True,
    #     "num_vis":       4,
    #     "vis_score_thr": 0.3,       # Ngưỡng cao hơn khi hiển thị
    #     "save_vis_dir":  "eval_vis", # None → plt.show() trực tiếp
    # }

    # ── Chạy evaluation ──────────────────────────────────────────
    results = evaluate_sbd(
        model_path   = CONFIG["model_path"],
        data_root    = CONFIG["data_root"],
        num_classes  = CONFIG["num_classes"],
        score_thresh = CONFIG["score_thresh"],
        verbose      = True,
    )

    # ── Visualization ─────────────────────────────────────────────
    if CONFIG["visualize"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = MVP_Seg(
            model_name     = "nvidia/MambaVision-T-1K",
            pretrained     = False,
            num_classes    = CONFIG["num_classes"],
            num_prototypes = 32,
        ).to(device)

        ckpt = torch.load(CONFIG["model_path"], map_location=device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)

        _, val_loader = get_sbd_dataloaders(
            root        = CONFIG["data_root"],
            batch_size  = 1,
            num_workers = 2,
            img_size    = IMG_SIZE,
            verbose     = False,
        )

        visualize_predictions(
            model        = model,
            val_loader   = val_loader,
            device       = device,
            num_samples  = CONFIG["num_vis"],
            score_thresh = CONFIG["vis_score_thr"],
            save_dir     = CONFIG.get("save_vis_dir", None),
            zip_path     = CONFIG.get("zip_path", None),
        )