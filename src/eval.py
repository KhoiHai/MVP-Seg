import torch
import torch.nn.functional as F
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import nms

from src.models.mvp_seg import MVP_Seg
from src.dataset.coco_dataset import get_dataloaders
from src.utils.flatten_predictions import flatten_predictions


def decode_predictions(outputs, top_k=200, nms_thresh=0.5, mask_thresh=0.5):

    cls_preds  = flatten_predictions(outputs["cls"])
    box_preds  = flatten_predictions(outputs["box"])
    coef_preds = flatten_predictions(outputs["coef"])
    proto      = outputs["proto"]

    B = cls_preds.shape[0]
    results = []

    for i in range(B):
        # Remove background (class 0)
        probs = cls_preds[i].softmax(-1)
        probs[:, 0] = 0  

        scores, labels = probs.max(-1)

        # Top-k
        k = min(top_k, scores.shape[0])
        topk_scores, topk_idx = scores.topk(k)

        boxes  = box_preds[i][topk_idx]
        labels = labels[topk_idx]
        coefs  = coef_preds[i][topk_idx]

        # NMS
        keep = nms(boxes, topk_scores, nms_thresh)

        boxes  = boxes[keep]
        labels = labels[keep]
        coefs  = coefs[keep]
        scores = topk_scores[keep]

        # Mask reconstruction
        proto_i = proto[i]
        Ph, Pw  = proto_i.shape[1], proto_i.shape[2]

        proto_flat = proto_i.permute(1, 2, 0).reshape(-1, proto_i.shape[0])
        pred_masks = torch.sigmoid(proto_flat @ coefs.T)
        pred_masks = pred_masks.reshape(Ph, Pw, -1).permute(2, 0, 1)

        pred_masks = pred_masks > mask_thresh

        results.append({
            "boxes":  boxes,
            "scores": scores,
            "labels": labels,
            "masks":  pred_masks
        })

    return results

@torch.no_grad()
def evaluate(model_path, data_root, num_classes=80, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on: {device}")

    # Model
    model = MVP_Seg(
        model_name="nvidia/MambaVision-T-1K",
        num_classes=num_classes,
        num_prototypes=32
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Loaded weights from: {model_path}")

    _, val_loader = get_dataloaders(
        data_root=data_root,
        batch_size=1,
        num_workers=2,
        subset_size=None
    )

    coco_gt = COCO(f"{data_root}/annotations/instances_val2017.json")
    coco_preds = []
    fps_list = []

    for batch in val_loader:
        if batch is None:
            continue

        images, targets = batch
        images = images.to(device)

        new_targets = []
        for t in targets:
            new_t = {}
            for k, v in t.items():
                new_t[k] = v.to(device) if isinstance(v, torch.Tensor) else v
            new_targets.append(new_t)
        targets = new_targets

        # FPS timing
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        outputs = model(images)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - t0
        fps_list.append(1.0 / elapsed if elapsed > 0 else 0)

        detections = decode_predictions(outputs)

        for det, tgt in zip(detections, targets):
            img_id = tgt["img_id"]
            if isinstance(img_id, torch.Tensor):
                img_id = int(img_id.item())

            boxes  = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()
            labels = det["labels"].cpu().numpy()

            for j in range(len(scores)):
                x1, y1, x2, y2 = boxes[j]

                coco_preds.append({
                    "image_id": img_id,
                    "category_id": int(labels[j]) + 1,
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1)
                    ],
                    "score": float(scores[j])
                })

    print("\n===== DETECTION METRICS (BBox) =====")

    if len(coco_preds) > 0:
        coco_dt   = coco_gt.loadRes(coco_preds)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    else:
        print("No predictions generated")

    print(f"\n===== SPEED =====")
    print(f"Average FPS : {np.mean(fps_list):.1f}")
    print(f"Median  FPS : {np.median(fps_list):.1f}")


if __name__ == "__main__":
    evaluate(
        model_path="checkpoints/best.pth",
        data_root="data/coco",
        num_classes=80
    )