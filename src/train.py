import os
import torch
from torch.optim import AdamW
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
from src.models.mvp_seg import MVP_Seg
from src.dataset.coco_dataset import get_coco_dataloaders
from src.dataset.sbd_dataset import get_sbd_dataloaders
from src.models.loss import Model_Loss
from src.eval import decode_predictions
import torch.nn.functional as F


def move_targets_to_device(targets, device):
    new_targets = []
    for t in targets:
        new_t = {}
        for k, v in t.items():
            new_t[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        new_targets.append(new_t)
    return new_targets

# -------------------------
# OPTIMIZER BUILDER
# -------------------------
def build_optimizer(model, base_lr, weight_decay, backbone_lr_ratio=0.5):
    return AdamW([
        {"params": model.backbone.parameters(), "lr": base_lr * backbone_lr_ratio},
        {"params": model.neck.parameters(), "lr": base_lr},
        {"params": model.proto.parameters(), "lr": base_lr},
        {"params": model.pred_head.parameters(), "lr": base_lr},
    ], weight_decay=weight_decay)

# -------------------------
# POLY LR
# -------------------------
def poly_lr_scheduler(optimizer, base_lrs, curr_iter, max_iter, power=1.0):
    factor = (1 - curr_iter / max_iter) ** power
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = base_lrs[i] * factor

# -------------------------
# EVALUATE mAP
# -------------------------
def evaluate_mAP(model, val_loader, device, data_root, img_size=550): # Đã thêm img_size
    model.eval()
    ann_file = os.path.join(data_root, "annotations/instances_val2017.json")
    coco_gt = COCO(ann_file)
    cat_ids = coco_gt.getCatIds()
    label_to_cat_id = {idx: cat_id for idx, cat_id in enumerate(cat_ids)}
    
    coco_preds = []
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None: continue
            images, targets = batch
            images = images.to(device)
            img_id = targets[0]["img_id"]
            if isinstance(img_id, torch.Tensor): img_id = int(img_id.item())
            
            img_info = coco_gt.loadImgs(img_id)[0]
            orig_h, orig_w = img_info['height'], img_info['width']
            
            outputs = model(images)
            detections = decode_predictions(outputs)[0]
            
            boxes = detections["boxes"].cpu().numpy()
            scores = detections["scores"].cpu().numpy()
            labels = detections["labels"].cpu().numpy()
            masks = detections["masks"].cpu()
            
            # Tính tỷ lệ quy đổi 1 LẦN DUY NHẤT
            scale_x = orig_w / img_size
            scale_y = orig_h / img_size
            
            for j in range(len(scores)):
                x1, y1, x2, y2 = boxes[j]
                
                # Nhân trực tiếp, không khai báo lại biến scale
                x1 = x1 * scale_x
                x2 = x2 * scale_x
                y1 = y1 * scale_y
                y2 = y2 * scale_y
                
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0: continue
                
                mask_tensor = masks[j].unsqueeze(0).unsqueeze(0).float()
                orig_mask = F.interpolate(mask_tensor, size=(orig_h, orig_w), mode='nearest').squeeze().numpy()
                orig_mask = orig_mask.astype(np.uint8)
                
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
        model.train()
        return 0.0
        
    coco_dt = coco_gt.loadRes(coco_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm") 
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    model.train() 
    return coco_eval.stats[0]
# -------------------------
# TRAIN
# -------------------------
def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # -------------------------
    # DATA
    # -------------------------
    # train_loader, val_loader = get_sbd_dataloaders(
    #     root=config["data_root"],
    #     batch_size=config["batch_size"],
    #     num_workers=config["num_workers"]
    # )

    train_loader, val_loader = get_coco_dataloaders(
        data_root=config["data_root"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        subset_size=config.get("subset_size", 10000) # Subset COCO 10k
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # -------------------------
    # MODEL
    # -------------------------
    model = MVP_Seg(
        model_name=config["backbone"],
        num_classes=config["num_classes"],
        num_prototypes=config["num_prototypes"]
    ).to(device)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    model.backbone.eval()
    print(f"Backbone frozen for {config['warmup_epochs']} epochs")

    # -------------------------
    # OPTIMIZER
    # -------------------------
    optimizer = build_optimizer(
        model,
        base_lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    base_lrs = [g["lr"] for g in optimizer.param_groups]

    criterion = Model_Loss(num_classes=config["num_classes"], alpha_box=1.5,)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    total_iters = config["epochs"] * len(train_loader)
    warmup_iters = 1500

    start_epoch = 0
    best_mAP = 0.0

    # -------------------------
    # FINETUNE TỪ PRETRAINED (Chuẩn bị cho bước qua SBD)
    # -------------------------
    if config.get("finetune", False) and not config.get("resume", False):
        path = config.get("finetune_path", "")
        if os.path.exists(path):
            print(f" Đang tải trọng số pretrained từ {path}")
            checkpoint = torch.load(path, map_location=device)
            state_dict = checkpoint.get("model_state", checkpoint)
            
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items() 
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f" Đã tải thành công {len(pretrained_dict)}/{len(model_dict)} layers.")

    # -------------------------
    # RESUME
    # -------------------------
    if config.get("resume", False):
        path = config.get("resume_path", "")
        if os.path.exists(path):
            print(f" Resuming from {path}")

            checkpoint = torch.load(path, map_location=device)

            model.load_state_dict(checkpoint["model_state"])

            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                # Override weight_decay mới
                for g in optimizer.param_groups:
                    g["weight_decay"] = config["weight_decay"]

            if "scaler_state" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state"])

            start_epoch = checkpoint.get("epoch", 0) + 1
            best_mAP = checkpoint.get("best_mAP", 0.0)

            print(f" Resume at epoch {start_epoch}")

            # nếu đã qua warmup → unfreeze luôn
            if start_epoch >= config["warmup_epochs"]:
                print("Resume after warmup → Unfreezing backbone")

                for param in model.backbone.parameters():
                    param.requires_grad = True

                optimizer = build_optimizer(
                    model,
                    base_lr=config["lr"],
                    weight_decay=config["weight_decay"]
                )

                base_lrs = [g["lr"] for g in optimizer.param_groups]

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    for epoch in range(start_epoch, config["epochs"]):

        # UNFREEZE đúng thời điểm
        if epoch == config["warmup_epochs"]:
            print(f"\nEpoch {epoch+1}: Unfreezing backbone")

            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer = build_optimizer(
                model,
                base_lr=config["lr"],
                weight_decay=config["weight_decay"]
            )

            base_lrs = [g["lr"] for g in optimizer.param_groups]

        model.train()
        if epoch < config["warmup_epochs"]:
            model.backbone.eval()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):

            if batch is None:
                continue

            images, targets = batch
            images = images.to(device)
            targets = move_targets_to_device(targets, device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(images)

            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]
            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=10.0
            )

            scaler.step(optimizer)
            scaler.update()

            # -------------------------
            # LR SCHEDULING
            # -------------------------
            global_iter = epoch * len(train_loader) + batch_idx

            if global_iter < warmup_iters:
                warmup_factor = global_iter / warmup_iters
                for i, g in enumerate(optimizer.param_groups):
                    g["lr"] = base_lrs[i] * warmup_factor
            else:
                poly_lr_scheduler(
                    optimizer,
                    base_lrs,
                    global_iter,
                    total_iters,
                    power=1.0
                )

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{config['epochs']}] "
                    f"Step [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(cls: {loss_dict['loss_cls'].item():.3f} "
                    f"box: {loss_dict['loss_box'].item():.3f} "
                    f"mask: {loss_dict['loss_mask'].item():.3f})"
                )

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"\nEpoch {epoch+1} | Train Loss: {avg_loss:.4f}")

        # -------------------------
        # SAVE
        # -------------------------
        os.makedirs(config["save_dir"], exist_ok=True)

        if (epoch + 1) % 10 == 0:
            # Tạo dictionary checkpoint chung
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_mAP": best_mAP 
            }

            # Lưu file checkpoint theo tên epoch 
            torch.save(checkpoint, os.path.join(config["save_dir"], f"epoch{epoch+1}.pth"))
            print(f" Đã lưu checkpoint: epoch_{epoch+1}.pth")

            print(f" Đang đánh giá mAP trên tập Validation (Epoch {epoch+1})...")
            current_mAP = evaluate_mAP(model, val_loader, device, config["data_root"])
            print(f" mAP hiện tại: {current_mAP:.4f} (Best: {best_mAP:.4f})")

            # Kiểm tra xem có vượt kỷ lục mAP không
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                checkpoint["best_mAP"] = best_mAP # Cập nhật lại mAP cao nhất vào dict
                
                # Lưu đè file best.pth
                torch.save(checkpoint, os.path.join(config["save_dir"], "best.pth"))
                print(f" Đã lưu Best Model mới tại epoch {epoch+1} với mAP: {best_mAP:.4f}!")

# ------------------------------
# CONFIG
# ------------------------------
if __name__ == "__main__":
    config = {
        "data_root": "/kaggle/input/coco2017",
        "save_dir": "/kaggle/working/checkpoints/coco",
        "subset_size": 10000, # Lấy 10k ảnh train

        "backbone": "nvidia/MambaVision-T-1K",
        "num_classes": 80,
        "num_prototypes": 32,

        "batch_size": 8,
        "num_workers": 2,

        "lr": 1e-4,
        "weight_decay": 0.01,

        "epochs": 100,
        "warmup_epochs": 5,

        "resume": False,
        "resume_path": "/kaggle/working/checkpoints/coco/last.pth",

        # Đặt là False khi train trên COCO.
        # Khi chuyển sang train SBD, đổi thành True và chỉ đường dẫn vào 'finetune_path'
        "finetune": False, 
        "finetune_path": "",
    }

    train(config)