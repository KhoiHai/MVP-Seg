import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from torchvision.ops import nms

from src.models.mvp_seg import MVP_Seg
from src.dataset.coco_dataset import get_coco_dataloaders, COCODataset
from src.utils.flatten_predictions import flatten_predictions
from src.utils.generate_locations import generate_locations

TOP_K = 200
NMS_THRESH = 0.5
STRIDES = [8, 16, 32]
IMG_SIZE = 550
COLORS = [
    (244, 67, 54), (233, 30, 99), (156, 39, 176), (103, 58, 183), (63, 81, 181),
    (33, 150, 243), (3, 169, 244), (0, 188, 212), (0, 150, 136), (76, 175, 80),
    (139, 195, 74), (205, 220, 57), (255, 235, 59), (255, 193, 7), (255, 152, 0),
    (255, 87, 34), (121, 85, 72), (158, 158, 158), (96, 125, 139), (0, 0, 0),
]


def _empty_result(device, hp, wp):
    return {
        "boxes_xyxy_550": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "scores": torch.zeros((0,), dtype=torch.float32, device=device),
        "labels": torch.zeros((0,), dtype=torch.long, device=device),
        "masks_550": torch.zeros((0, IMG_SIZE, IMG_SIZE), dtype=torch.float32, device=device),
        "masks_proto": torch.zeros((0, hp, wp), dtype=torch.float32, device=device),
    }


def decode_predictions(outputs, top_k=TOP_K, nms_thresh=NMS_THRESH):
    cls_preds = flatten_predictions(outputs["cls"])    # [B, N, C]
    box_preds = flatten_predictions(outputs["box"])    # [B, N, 4] ltrb_norm
    coef_preds = flatten_predictions(outputs["coef"])  # [B, N, P]
    proto = outputs["proto"]                           # [B, P, Hp, Wp]
    locations = generate_locations(outputs["cls"], STRIDES).to(cls_preds.device)  # [N,2] float32

    B = cls_preds.shape[0]
    results = []

    for i in range(B):
        hp, wp = int(proto.shape[2]), int(proto.shape[3])

        # sigmoid (not softmax)
        cls_scores = torch.sigmoid(cls_preds[i])  # [N, C]
        num_classes = cls_scores.shape[1]

        # per-class candidates + NMS
        all_keep_boxes = []
        all_keep_scores = []
        all_keep_labels = []
        all_keep_coefs = []
        all_keep_locs = []

        for c in range(num_classes):
            scores_c = cls_scores[:, c]
            k = min(top_k, scores_c.numel())
            if k <= 0:
                continue
            top_scores, top_idx = scores_c.topk(k)
            if top_scores.numel() == 0:
                continue

            boxes_ltrb = box_preds[i][top_idx]       # [k,4] ltrb_norm
            coefs_c = coef_preds[i][top_idx]
            locs_c = locations[top_idx]              # [k,2]

            # decode symmetry with loss:
            # ltrb = [px-x1, py-y1, x2-px, y2-py] / IMG_SIZE
            # => x1=px-l*IMG_SIZE, ...
            l = boxes_ltrb[:, 0] * IMG_SIZE
            t = boxes_ltrb[:, 1] * IMG_SIZE
            r = boxes_ltrb[:, 2] * IMG_SIZE
            b = boxes_ltrb[:, 3] * IMG_SIZE
            px = locs_c[:, 0]
            py = locs_c[:, 1]

            x1 = (px - l).clamp(0, IMG_SIZE)
            y1 = (py - t).clamp(0, IMG_SIZE)
            x2 = (px + r).clamp(0, IMG_SIZE)
            y2 = (py + b).clamp(0, IMG_SIZE)
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

            valid = (x2 > x1) & (y2 > y1)
            if valid.sum() == 0:
                continue
            boxes_xyxy = boxes_xyxy[valid]
            top_scores = top_scores[valid]
            coefs_c = coefs_c[valid]
            locs_c = locs_c[valid]

            keep = nms(boxes_xyxy, top_scores, nms_thresh)
            if keep.numel() == 0:
                continue

            all_keep_boxes.append(boxes_xyxy[keep])
            all_keep_scores.append(top_scores[keep])
            all_keep_labels.append(torch.full((keep.numel(),), c, dtype=torch.long, device=cls_preds.device))
            all_keep_coefs.append(coefs_c[keep])
            all_keep_locs.append(locs_c[keep])

        if len(all_keep_boxes) == 0:
            results.append(_empty_result(cls_preds.device, hp, wp))
            continue

        boxes = torch.cat(all_keep_boxes, dim=0)
        scores = torch.cat(all_keep_scores, dim=0)
        labels = torch.cat(all_keep_labels, dim=0)
        coefs = torch.cat(all_keep_coefs, dim=0)

        # keep global top-k after per-class nms
        k_final = min(top_k, scores.numel())
        top_scores, ord_idx = scores.topk(k_final)
        boxes = boxes[ord_idx]
        labels = labels[ord_idx]
        coefs = coefs[ord_idx]
        scores = top_scores

        # mask sigmoid(coef @ proto^T)
        p = proto.shape[1]
        proto_flat = proto[i].view(p, -1).T                 # [hp*wp, p]
        mask_logits = proto_flat @ coefs.T                  # [hp*wp, k]
        masks_proto = torch.sigmoid(mask_logits.T.view(-1, hp, wp))

        # crop mask in box region (prototype space)
        scale_x = wp / float(IMG_SIZE)
        scale_y = hp / float(IMG_SIZE)
        boxes_proto = boxes.clone()
        boxes_proto[:, [0, 2]] *= scale_x
        boxes_proto[:, [1, 3]] *= scale_y

        y_grid, x_grid = torch.meshgrid(
            torch.arange(hp, device=cls_preds.device),
            torch.arange(wp, device=cls_preds.device),
            indexing="ij",
        )
        xg = x_grid.unsqueeze(0)
        yg = y_grid.unsqueeze(0)
        crop = (
            (xg >= boxes_proto[:, 0].view(-1, 1, 1)) &
            (xg <= boxes_proto[:, 2].view(-1, 1, 1)) &
            (yg >= boxes_proto[:, 1].view(-1, 1, 1)) &
            (yg <= boxes_proto[:, 3].view(-1, 1, 1))
        ).float()
        masks_proto = masks_proto * crop

        # upsample to 550x550 (model input space)
        masks_550 = F.interpolate(
            masks_proto.unsqueeze(1),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        results.append({
            "boxes_xyxy_550": boxes,
            "scores": scores,
            "labels": labels,
            "masks_550": masks_550,
            "masks_proto": masks_proto,
        })

    return results


def _resize_boxes_xyxy(boxes_xyxy, resized_size, orig_size):
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    rh, rw = resized_size
    oh, ow = orig_size
    sx = float(ow) / float(rw)
    sy = float(oh) / float(rh)
    out = boxes_xyxy.clone()
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    out[:, 0::2] = out[:, 0::2].clamp(0, ow)
    out[:, 1::2] = out[:, 1::2].clamp(0, oh)
    return out


def _resize_masks_to_orig(masks_550, orig_size):
    if masks_550.numel() == 0:
        return masks_550
    oh, ow = orig_size
    return F.interpolate(
        masks_550.unsqueeze(1),
        size=(oh, ow),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _encode_binary_mask(mask_np):
    rle = mask_utils.encode(np.asfortranarray(mask_np))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


@torch.no_grad()
def evaluate_coco(
    model_path,
    data_root,
    num_classes=80,
    num_prototypes=32,
    img_size=IMG_SIZE,
    top_k=TOP_K,
    nms_thresh=NMS_THRESH,
    score_thresh=0.05,
    max_images=None,
    visualize=False,
    vis_dir="vis_coco",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on: {device}")

    model = MVP_Seg(
        model_name="nvidia/MambaVision-T-1K",
        pretrained=False,
        num_classes=num_classes,
        num_prototypes=num_prototypes,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    _, val_loader = get_coco_dataloaders(
        data_root=data_root,
        batch_size=1,
        num_workers=2,
        subset_size=None,
        img_size=img_size,
        val_subset_size=max_images,
    )
    val_dataset = val_loader.dataset
    if not isinstance(val_dataset, COCODataset):
        raise RuntimeError("Expected COCODataset for val loader.")
    label_to_cat_id = val_dataset.label_to_cat_id

    ann_file = os.path.join(data_root, "annotations/instances_val2017.json")
    coco_gt = COCO(ann_file)

    bbox_results = []
    segm_results = []
    fps_list = []

    if visualize:
        os.makedirs(vis_dir, exist_ok=True)

    for batch_idx, batch in enumerate(val_loader):
        if batch is None:
            continue
        images, targets = batch
        images = images.to(device, non_blocking=True)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(images)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        if dt > 0:
            fps_list.append(1.0 / dt)

        detections = decode_predictions(outputs, top_k=top_k, nms_thresh=nms_thresh)
        det = detections[0]
        tgt = targets[0]
        img_id = int(tgt["img_id"])
        orig_size = tuple(tgt["orig_size"])
        resized_size = tuple(tgt["resized_size"])

        keep = det["scores"] >= score_thresh
        boxes_550 = det["boxes_xyxy_550"][keep]
        scores = det["scores"][keep]
        labels = det["labels"][keep]
        masks_550 = det["masks_550"][keep]

        boxes_orig = _resize_boxes_xyxy(boxes_550, resized_size, orig_size)
        masks_orig = _resize_masks_to_orig(masks_550, orig_size)
        masks_bin = (masks_orig >= 0.5).to(torch.uint8).cpu().numpy()

        boxes_np = boxes_orig.cpu().numpy()
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for j in range(scores_np.shape[0]):
            x1, y1, x2, y2 = boxes_np[j].tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cat_id = int(label_to_cat_id[int(labels_np[j])])

            bbox_results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(scores_np[j]),
            })
            segm_results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": _encode_binary_mask(masks_bin[j]),
                "score": float(scores_np[j]),
            })

        if visualize and scores_np.shape[0] > 0:
            file_name = coco_gt.loadImgs([img_id])[0]["file_name"]
            image_path = os.path.join(data_root, "val2017", file_name)
            img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
            overlay = img.copy()
            for j in range(scores_np.shape[0]):
                color = np.array(COLORS[j % 20], dtype=np.float32)
                m = masks_bin[j].astype(bool)
                overlay[m] = overlay[m] * 0.5 + color * 0.5
            vis = np.clip(overlay, 0, 255).astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(vis_dir, f"{img_id}.png"))

    metrics = {}

    if len(bbox_results) > 0:
        coco_dt_bbox = coco_gt.loadRes(bbox_results)
        eval_bbox = COCOeval(coco_gt, coco_dt_bbox, "bbox")
        eval_bbox.evaluate()
        eval_bbox.accumulate()
        eval_bbox.summarize()
        metrics["bbox_mAP_50_95"] = float(eval_bbox.stats[0])
        metrics["bbox_AP50"] = float(eval_bbox.stats[1])
        metrics["bbox_AP75"] = float(eval_bbox.stats[2])
        metrics["bbox_APS"] = float(eval_bbox.stats[3])
        metrics["bbox_APM"] = float(eval_bbox.stats[4])
        metrics["bbox_APL"] = float(eval_bbox.stats[5])
    else:
        print("No bbox predictions to evaluate.")

    if len(segm_results) > 0:
        coco_dt_segm = coco_gt.loadRes(segm_results)
        eval_segm = COCOeval(coco_gt, coco_dt_segm, "segm")
        eval_segm.evaluate()
        eval_segm.accumulate()
        eval_segm.summarize()
        metrics["segm_mAP_50_95"] = float(eval_segm.stats[0])
        metrics["segm_AP50"] = float(eval_segm.stats[1])
        metrics["segm_AP75"] = float(eval_segm.stats[2])
        metrics["segm_APS"] = float(eval_segm.stats[3])
        metrics["segm_APM"] = float(eval_segm.stats[4])
        metrics["segm_APL"] = float(eval_segm.stats[5])
    else:
        print("No segm predictions to evaluate.")

    fps = float(np.mean(fps_list)) if len(fps_list) > 0 else 0.0
    metrics["FPS"] = fps
    print(f"FPS: {fps:.2f}")
    return metrics


if __name__ == "__main__":
    evaluate_coco(
        model_path="checkpoints_coco/best.pth",
        data_root="/content/coco",
        num_classes=80,
        num_prototypes=32,
        img_size=IMG_SIZE,
        top_k=TOP_K,
        nms_thresh=NMS_THRESH,
        score_thresh=0.05,
        max_images=None,
        visualize=False,
    )
