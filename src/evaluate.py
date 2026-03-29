import torch
import torch.nn.functional as F
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import nms

from src.models.mvp_seg import MVPSeg
from src.dataset import get_dataloaders


def flatten_predictions(feature_list):
    '''
    Flatten multi-scale prediction tensors into a single sequence
        Converts list of [B, C, Hi, Wi] -> [B, sum(Hi*Wi), C]
    Args:
        feature_list: list of Tensors [B, C, Hi, Wi]
    Returns:
        Tensor [B, total_anchors, C]
    '''
    out = []
    for f in feature_list:
        B, C, H, W = f.shape
        out.append(f.permute(0, 2, 3, 1).reshape(B, -1, C))
    return torch.cat(out, dim = 1)


def decode_predictions(outputs, top_k = 200, nms_thresh = 0.5, mask_thresh = 0.5):
    '''
    Post-process raw model outputs into final detections
        Steps: flatten -> top-k selection -> NMS -> mask reconstruction -> upsample
    Args:
        outputs     : dict from MVPSeg.forward() with cls, box, coef, proto
        top_k       : Maximum number of candidates to keep before NMS (200 per proposal)
        nms_thresh  : IoU threshold for Non-Maximum Suppression (0.5 per proposal)
        mask_thresh : Sigmoid threshold to binarize reconstructed masks
    Returns:
        list of dicts, one per image:
            boxes  : [K, 4]  final bounding boxes
            scores : [K]     confidence scores
            labels : [K]     predicted class indices (0-based)
            masks  : [K, H, W] binary instance masks (at prototype resolution)
    '''
    cls_preds  = flatten_predictions(outputs["cls"])   # [B, #pred, num_classes]
    box_preds  = flatten_predictions(outputs["box"])   # [B, #pred, 4]
    coef_preds = flatten_predictions(outputs["coef"])  # [B, #pred, P]
    proto      = outputs["proto"]                      # [B, P, H/4, W/4]

    B = cls_preds.shape[0]
    results = []

    for i in range(B):
        # Get class score and label per anchor (ignore background at index 0 if applicable)
        scores, labels = cls_preds[i].softmax(-1).max(-1)  # [#pred], [#pred]

        # Top-k selection to reduce computation before NMS
        k = min(top_k, scores.shape[0])
        topk_scores, topk_idx = scores.topk(k)

        boxes  = box_preds[i][topk_idx]    # [k, 4]
        labels = labels[topk_idx]          # [k]
        coefs  = coef_preds[i][topk_idx]   # [k, P]

        # Non-Maximum Suppression to remove duplicate detections
        keep   = nms(boxes, topk_scores, nms_thresh)
        boxes  = boxes[keep]
        labels = labels[keep]
        coefs  = coefs[keep]
        scores = topk_scores[keep]

        # Reconstruct instance masks: M = sigmoid(proto @ coef^T)
        proto_i    = proto[i]                                           # [P, H/4, W/4]
        Ph, Pw     = proto_i.shape[1], proto_i.shape[2]
        proto_flat = proto_i.permute(1, 2, 0).reshape(-1, proto_i.shape[0])
        # [H/4*W/4, P] @ [P, K] -> [H/4*W/4, K]
        pred_masks = torch.sigmoid(proto_flat @ coefs.T)
        pred_masks = pred_masks.reshape(Ph, Pw, -1).permute(2, 0, 1)   # [K, H/4, W/4]

        # Binarize masks at threshold
        pred_masks = pred_masks > mask_thresh                           # bool [K, H/4, W/4]

        results.append({
            "boxes":  boxes,
            "scores": scores,
            "labels": labels,
            "masks":  pred_masks
        })

    return results


@torch.no_grad()
def evaluate(model_path, data_root, num_classes = 80, device = None):
    '''
    Evaluate MVP-Seg on COCO val subset
        Computes COCO-standard mAP metrics (AP, AP50, AP75, APS, APM, APL)
        and measures inference FPS on the validation set
    Args:
        model_path: Path to trained weights (.pth file)
        data_root : Root path of COCO dataset
        num_classes: Number of object classes
        device    : "cuda" or "cpu" (auto-detected if None)
    '''
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on: {device}")

    # Load model
    model = MVPSeg(num_classes = num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()
    print(f"Loaded weights from: {model_path}")

    # Val loader (batch_size=1 for accurate per-image evaluation)
    _, val_loader = get_dataloaders(
        data_root  = data_root,
        batch_size = 1,
        num_workers = 2,
        subset_size = None
    )

    coco_gt    = COCO(f"{data_root}/annotations/instances_val2017.json")
    coco_preds = []   # List of COCO-format prediction dicts
    fps_list   = []   # FPS per image

    for batch in val_loader:
        if batch is None:
            continue
        images, targets = batch
        images = images.to(device)

        # Measure inference time per image
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        outputs = model(images)

        if device == "cuda":
            torch.cuda.synchronize()
        fps_list.append(1.0 / (time.time() - t0 + 1e-9))

        # Decode predictions to final detections
        detections = decode_predictions(outputs)

        for det, tgt in zip(detections, targets):
            img_id = tgt["img_id"]
            boxes  = det["boxes"].cpu().numpy()    # [K, 4]  x1,y1,x2,y2
            scores = det["scores"].cpu().numpy()   # [K]
            labels = det["labels"].cpu().numpy()   # [K]  0-based

            for j in range(len(scores)):
                x1, y1, x2, y2 = boxes[j]
                coco_preds.append({
                    "image_id":    img_id,
                    "category_id": int(labels[j]) + 1,    # COCO uses 1-based category IDs
                    "bbox":        [float(x1), float(y1),
                                    float(x2 - x1), float(y2 - y1)],  # COCO [x,y,w,h] format
                    "score":       float(scores[j])
                })

    # Compute COCO mAP using official pycocotools evaluator
    print("\n===== DETECTION METRICS (BBox) =====")
    if len(coco_preds) > 0:
        coco_dt   = coco_gt.loadRes(coco_preds)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # coco_eval.stats: [mAP, AP50, AP75, APS, APM, APL, AR1, AR10, AR100, ARS, ARM, ARL]
    else:
        print("No predictions generated - check model weights or threshold settings")

    # Report FPS
    print(f"\n===== SPEED =====")
    print(f"Average FPS : {np.mean(fps_list):.1f}")
    print(f"Median  FPS : {np.median(fps_list):.1f}")


# ------------------------------
# EVALUATION TESTING
# ------------------------------
if __name__ == "__main__":
    evaluate(
        model_path  = "checkpoints/best.pth",
        data_root   = "data/coco",
        num_classes = 80
    )