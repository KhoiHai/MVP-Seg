import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
from torchvision.ops import batched_nms
import json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.mvp_seg import MVP_Seg
from src.dataset.coco_dataset import get_coco_dataloaders
from src.utils.flatten_predictions import flatten_predictions
from src.utils.generate_locations import generate_locations

def decode_predictions(outputs, strides=[8, 16, 32], img_size=550, score_thresh=0.05, top_k=200, nms_thresh=0.5):
    cls_preds  = flatten_predictions(outputs["cls"])    
    box_preds  = flatten_predictions(outputs["box"])    
    coef_preds = flatten_predictions(outputs["coef"])   
    proto      = outputs["proto"]                       

    locations = generate_locations(outputs["cls"], strides).to(cls_preds.device) 

    # ----- KHÔI PHỤC STRIDE -----
    level_sizes = [x.shape[2] * x.shape[3] for x in outputs["cls"]]
    stride_tensor = []
    for size, s_val in zip(level_sizes, strides):
        stride_tensor.append(torch.full((int(size),), float(s_val), device=cls_preds.device))
    stride_tensor = torch.cat(stride_tensor)

    B = cls_preds.shape[0]
    results = []

    for i in range(B):
        scores_all = torch.sigmoid(cls_preds[i])        
        scores, labels = scores_all.max(dim=-1)         

        keep = scores > score_thresh
        if keep.sum() == 0:
            results.append({"boxes": [], "scores": [], "labels": [], "masks": []})
            continue

        scores_f = scores[keep]
        labels_f = labels[keep]
        coefs_f  = coef_preds[i][keep]     
        locs_f   = locations[keep]         

        # ----- NHÂN NGƯỢC STRIDE VÀO BOX -----
        pos_strides = stride_tensor[keep].unsqueeze(1)
        boxes_f  = box_preds[i][keep] * pos_strides

        k = min(top_k, scores_f.shape[0])
        topk_scores, topk_idx = scores_f.topk(k)

        boxes_f  = boxes_f[topk_idx]
        labels_f = labels_f[topk_idx]
        coefs_f  = coefs_f[topk_idx]
        locs_f   = locs_f[topk_idx]

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

        xyxy_boxes = torch.stack([x1, y1, x2, y2], dim=1) 

        keep_final = batched_nms(xyxy_boxes, topk_scores, labels_f, nms_thresh)

        if len(keep_final) == 0:
            results.append({"boxes": [], "scores": [], "labels": [], "masks": []})
            continue

        # Giới hạn tối đa 100 detection theo chuẩn COCO (maxDets=100)
        keep_final = keep_final[:100]
        final_boxes  = xyxy_boxes[keep_final]
        final_labels = labels_f[keep_final]
        final_scores = topk_scores[keep_final]
        final_coefs  = coefs_f[keep_final]

        P, Hp, Wp  = proto.shape[1:]
        proto_flat = proto[i].view(P, -1).T          
        mask_logits = proto_flat @ final_coefs.T      
        mask_logits = mask_logits.T.view(-1, Hp, Wp)  
        
        pred_masks = F.interpolate(mask_logits.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)
        pred_masks = torch.sigmoid(pred_masks)

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
            "masks":  pred_masks > 0.5 
        })

    return results

@torch.no_grad()
def evaluate(model_path, data_root, num_classes=80, device=None, img_size=550):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Bắt đầu đánh giá trên: {device}")

    model = MVP_Seg(
        model_name="nvidia/MambaVision-T-1K",
        pretrained=False,
        num_classes=num_classes,
        num_prototypes=32
    ).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Đã tải trọng số từ: {model_path}")

    _, val_loader = get_coco_dataloaders(
        data_root=data_root, batch_size=1, num_workers=2, subset_size=None, img_size=img_size
    )

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

        # --- LẤY KÍCH THƯỚC GỐC ĐỂ RESIZE ---
        img_info = coco_gt.loadImgs(img_id)[0]
        orig_h, orig_w = img_info['height'], img_info['width']
        scale_x = orig_w / img_size
        scale_y = orig_h / img_size

        if device == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        
        outputs = model(images)
        
        if device == "cuda": torch.cuda.synchronize()
        elapsed = time.time() - t0
        fps_list.append(1.0 / elapsed if elapsed > 0 else 0)

        detections = decode_predictions(outputs, img_size=img_size)[0]
        boxes  = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()
        masks  = detections["masks"].cpu()

        for j in range(len(scores)):
            x1, y1, x2, y2 = boxes[j]
            
            # Khôi phục tọa độ hộp
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y
            
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0: continue
            
            # Khôi phục mặt nạ
            mask_tensor = masks[j].unsqueeze(0).unsqueeze(0).float()
            orig_mask_tensor = F.interpolate(mask_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze()
            orig_mask = (orig_mask_tensor > 0.5).cpu().numpy().astype(np.uint8)
            
            mask_rle = maskUtils.encode(np.asfortranarray(orig_mask))
            mask_rle['counts'] = mask_rle['counts'].decode('utf-8')

            coco_preds.append({
                "image_id": img_id,
                "category_id": label_to_cat_id[int(labels[j])], 
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "segmentation": mask_rle,
                "score": float(scores[j])
            })

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

@torch.no_grad()
def generate_test_dev_json(model_path, data_root, json_name="predictions_test_dev.json", num_classes=80, device=None, img_size=550):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Bắt đầu quá trình tạo JSON cho test-dev trên: {device}")

    # 1. Khởi tạo Model
    model = MVP_Seg(
        model_name="nvidia/MambaVision-T-1K",
        pretrained=False,
        num_classes=num_classes,
        num_prototypes=32
    ).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Đã tải trọng số từ: {model_path}")

    # 2. Nạp file info của test-dev (Chỉ có info, không có nhãn)
    ann_file = os.path.join(data_root, "annotations/image_info_test-dev2017.json")
    coco_test = COCO(ann_file)
    img_ids = coco_test.getImgIds()
    
    # Ánh xạ label (0-79) sang category_id của COCO (1-90)
    cat_ids = coco_test.getCatIds()
    label_to_cat_id = {idx: cat_id for idx, cat_id in enumerate(cat_ids)}

    # 3. Pipeline Transform thủ công cho ảnh test
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    test_dir = os.path.join(data_root, "test2017")
    coco_preds = []

    print(f"\n[INFO] Đang chạy Inference trên {len(img_ids)} ảnh test-dev...")
    
    for i, img_id in enumerate(img_ids):
        img_info = coco_test.loadImgs(img_id)[0]
        img_path = os.path.join(test_dir, img_info['file_name'])
        
        # Đọc ảnh gốc
        image_np = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = img_info['height'], img_info['width']
        
        # Tiền xử lý
        tensor_img = transform(image=image_np)["image"].unsqueeze(0).to(device)
        
        # Suy luận (Inference)
        outputs = model(tensor_img)
        detections = decode_predictions(outputs, img_size=img_size)[0]
        
        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()
        masks = detections["masks"].cpu()

        scale_x = orig_w / img_size
        scale_y = orig_h / img_size

        for j in range(len(scores)):
            x1, y1, x2, y2 = boxes[j]
            
            x1 *= scale_x; x2 *= scale_x
            y1 *= scale_y; y2 *= scale_y
            
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0: continue
            
            mask_tensor = masks[j].unsqueeze(0).unsqueeze(0).float()
            orig_mask_tensor = F.interpolate(mask_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze()
            orig_mask = (orig_mask_tensor > 0.5).cpu().numpy().astype(np.uint8)
            
            # Gọt sạch nhiễu ngoài box để điểm AP cao nhất
            crop_x1 = max(0, int(np.floor(x1)))
            crop_y1 = max(0, int(np.floor(y1)))
            crop_x2 = min(orig_w, int(np.ceil(x2)))
            crop_y2 = min(orig_h, int(np.ceil(y2)))
            orig_mask[:crop_y1, :] = 0
            orig_mask[crop_y2:, :] = 0
            orig_mask[:, :crop_x1] = 0
            orig_mask[:, crop_x2:] = 0
            
            if orig_mask.sum() == 0: continue
            
            # Mã hóa RLE chuẩn COCO
            mask_rle = maskUtils.encode(np.asfortranarray(orig_mask))
            mask_rle['counts'] = mask_rle['counts'].decode('utf-8')

            coco_preds.append({
                "image_id": img_id,
                "category_id": label_to_cat_id[int(labels[j])],
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "segmentation": mask_rle,
                "score": float(scores[j])
            })
            
        if (i + 1) % 1000 == 0:
            print(f" Đã xử lý {i + 1}/{len(img_ids)} ảnh...")

    # 4. Lưu JSON
    save_json_path = os.path.join(os.path.dirname(model_path), json_name)
    print(f"\n[INFO] Đang ghi {len(coco_preds)} dự đoán ra file JSON...")
    with open(save_json_path, "w") as f:
        json.dump(coco_preds, f)
    print(f" Đã lưu file thành công tại: {save_json_path}")

if __name__ == "__main__":
    evaluate(
        model_path="/kaggle/working/checkpoints/coco/best.pth", 
        data_root="/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017",
        num_classes=80
    )