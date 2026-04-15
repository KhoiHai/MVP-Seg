import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
from torchvision.ops import nms

from src.models.mvp_seg import MVP_Seg
from src.dataset.coco_dataset import get_coco_dataloaders
from src.utils.flatten_predictions import flatten_predictions
from src.utils.generate_locations import generate_locations


def decode_predictions(outputs, strides=[8, 16, 32], img_size=550, score_thresh=0.05, top_k=200, nms_thresh=0.5):
    cls_preds  = flatten_predictions(outputs["cls"])    # [B, N, C]
    box_preds  = flatten_predictions(outputs["box"])    # [B, N, 4]
    coef_preds = flatten_predictions(outputs["coef"])   # [B, N, P]
    proto      = outputs["proto"]                       # [B, P, Hp, Wp]

    locations = generate_locations(outputs["cls"], strides).to(cls_preds.device) # [N, 2]

    B = cls_preds.shape[0]
    results = []

    for i in range(B):
        # 1. Score: Dùng sigmoid vì train bằng Focal Loss
        scores_all = torch.sigmoid(cls_preds[i])        
        scores, labels = scores_all.max(dim=-1)         

        # 2. Lọc sơ bộ bằng score_thresh
        keep = scores > score_thresh
        if keep.sum() == 0:
            results.append({"boxes": [], "scores": [], "labels": [], "masks": []})
            continue

        scores_f = scores[keep]
        labels_f = labels[keep]
        boxes_f  = box_preds[i][keep]      
        coefs_f  = coef_preds[i][keep]     
        locs_f   = locations[keep]         

        # 3. Top-k
        k = min(top_k, scores_f.shape[0])
        topk_scores, topk_idx = scores_f.topk(k)

        boxes_f  = boxes_f[topk_idx]
        labels_f = labels_f[topk_idx]
        coefs_f  = coefs_f[topk_idx]
        locs_f   = locs_f[topk_idx]

        # 4. Chuyển box từ LTRB sang XYXY pixel 
        # (Đã bỏ chia img_size trong loss nên LTRB giờ là pixel thật)
        l = boxes_f[:, 0]
        t = boxes_f[:, 1]
        r = boxes_f[:, 2]
        b = boxes_f[:, 3]

        px = locs_f[:, 0].float()
        py = locs_f[:, 1].float()

        x1 = (px - l).clamp(0, img_size)
        y1 = (py - t).clamp(0, img_size)
        x2 = (px + r).clamp(0, img_size)
        y2 = (py + b).clamp(0, img_size)

        xyxy_boxes = torch.stack([x1, y1, x2, y2], dim=1)  # [K, 4]

        # 5. Fast NMS (Per-class)
        keep_final = []
        for cls_id in labels_f.unique():
            cls_mask   = labels_f == cls_id
            cls_boxes  = xyxy_boxes[cls_mask]
            cls_scores = topk_scores[cls_mask]
            keep_cls   = nms(cls_boxes, cls_scores, nms_thresh)
            global_idx = cls_mask.nonzero(as_tuple=True)[0]
            keep_final.append(global_idx[keep_cls])

        if len(keep_final) == 0:
            results.append({"boxes": [], "scores": [], "labels": [], "masks": []})
            continue

        keep_final   = torch.cat(keep_final)
        final_boxes  = xyxy_boxes[keep_final]
        final_labels = labels_f[keep_final]
        final_scores = topk_scores[keep_final]
        final_coefs  = coefs_f[keep_final]

        # 6. Tái tạo Mask: M = sigmoid(coef @ proto^T)
        P, Hp, Wp  = proto.shape[1:]
        proto_flat = proto[i].view(P, -1).T          
        mask_logits = proto_flat @ final_coefs.T      
        mask_logits = mask_logits.T.view(-1, Hp, Wp)  
        
        # Upsample mask lên kích thước ảnh gốc
        pred_masks = F.interpolate(mask_logits.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)
        pred_masks = torch.sigmoid(pred_masks)

        # Crop mask theo Bounding Box để bỏ viền nhiễu
        for j in range(len(final_boxes)):
            bx1, by1, bx2, by2 = map(int, final_boxes[j].clamp(0, img_size-1))
            crop_mask = torch.zeros_like(pred_masks[j])
            if bx2 > bx1 and by2 > by1:
                crop_mask[by1:by2, bx1:bx2] = pred_masks[j, by1:by2, bx1:bx2]
            pred_masks[j] = crop_mask

        results.append({
            "boxes":  final_boxes,
            "scores": final_scores,
            "labels": final_labels,
            "masks":  pred_masks > 0.5 # Binarize
        })

    return results


@torch.no_grad()
def evaluate(model_path, data_root, num_classes=80, device=None, img_size=550):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Bắt đầu đánh giá trên: {device}")

    # Khởi tạo mô hình
    model = MVP_Seg(
        model_name="nvidia/MambaVision-T-1K",
        pretrained=False,
        num_classes=num_classes,
        num_prototypes=32
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Đã tải trọng số từ: {model_path}")

    # Tải Dataloader (batch_size=1 để tính FPS chuẩn)
    _, val_loader = get_coco_dataloaders(
        data_root=data_root, batch_size=1, num_workers=2, subset_size=None, img_size=img_size
    )

    # Load COCO Ground Truth và tạo bộ ánh xạ Category ID
    ann_file = os.path.join(data_root, "annotations/instances_val2017.json")
    coco_gt = COCO(ann_file)
    cat_ids = coco_gt.getCatIds()
    label_to_cat_id = {idx: cat_id for idx, cat_id in enumerate(cat_ids)}

    coco_preds = []
    fps_list = []

    print(f"\n[INFO] Đang đánh giá {len(val_loader)} ảnh val...\n")

    for batch_idx, batch in enumerate(val_loader):
        if batch is None: continue

        images, targets = batch
        images = images.to(device)
        img_id = targets[0]["img_id"]
        if isinstance(img_id, torch.Tensor): img_id = int(img_id.item())

        # Tính FPS
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        
        outputs = model(images)
        
        if device == "cuda": torch.cuda.synchronize()
        elapsed = time.time() - t0
        fps_list.append(1.0 / elapsed if elapsed > 0 else 0)

        # Giải mã
        detections = decode_predictions(outputs, img_size=img_size)[0]
        boxes  = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()
        masks  = detections["masks"].cpu().numpy() # [K, H, W]

        for j in range(len(scores)):
            x1, y1, x2, y2 = boxes[j]
            w, h = x2 - x1, y2 - y1
            
            if w <= 0 or h <= 0: continue
            
            # Encode Mask theo chuẩn RLE của COCO
            mask_rle = maskUtils.encode(np.asfortranarray(masks[j].astype(np.uint8)))
            mask_rle['counts'] = mask_rle['counts'].decode('utf-8')

            coco_preds.append({
                "image_id": img_id,
                "category_id": label_to_cat_id[int(labels[j])], # Ánh xạ lại ID chuẩn
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "segmentation": mask_rle,
                "score": float(scores[j])
            })
            
        if (batch_idx + 1) % 200 == 0:
            print(f"  [{batch_idx + 1}/{len(val_loader)}] ảnh đã được xử lý...")

    if len(coco_preds) == 0:
        print("Mô hình không tạo ra dự đoán nào!")
        return

    print("\n===== ĐÁNH GIÁ BOUNDING BOX (BBox) =====")
    coco_dt = coco_gt.loadRes(coco_preds)
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()

    print("\n===== ĐÁNH GIÁ MẶT NẠ (Segmentation) =====")
    coco_eval_segm = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()

    print(f"\n===== TỐC ĐỘ (SPEED) =====")
    print(f"Average FPS : {np.mean(fps_list):.1f}")
    print(f"Median  FPS : {np.median(fps_list):.1f}")


if __name__ == "__main__":
    # Cấu hình đường dẫn cho Kaggle (Hoặc máy local)
    evaluate(
        model_path="/kaggle/working/checkpoints/coco/best.pth", # Thay đổi theo vị trí save
        data_root="/kaggle/input/coco2017",                     # Thay đổi theo dataset add vào
        num_classes=80
    )