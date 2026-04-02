import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.detection import MeanAveragePrecision

from src.models.mvp_seg import MVP_Seg
from src.dataset.sbd_dataset import get_sbd_dataloaders
from src.models.loss import Model_Loss


def move_targets_to_device(targets, device):
    new_targets = []
    for t in targets:
        new_t = {}
        for k, v in t.items():
            new_t[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        new_targets.append(new_t)
    return new_targets


# ─────────────────────────────────────────────────────────────
# Decode raw model outputs → boxes + scores + labels + masks
# ─────────────────────────────────────────────────────────────
def decode_outputs(outputs, score_thresh=0.3, img_size=550):
    """
    Convert raw model outputs to per-image detection lists
    for torchmetrics MeanAveragePrecision.

    Returns:
        preds  : list of dicts  { boxes, scores, labels, masks }
    """
    from src.utils.flatten_predictions import flatten_predictions
    from src.utils.generate_locations   import generate_locations
    import torch.nn.functional as F

    cls_preds  = flatten_predictions(outputs["cls"])   # [B, N, C]
    box_preds  = flatten_predictions(outputs["box"])   # [B, N, 4]
    coef_preds = flatten_predictions(outputs["coef"])  # [B, N, P]
    proto      = outputs["proto"]                       # [B, P, Hp, Wp]

    locations  = generate_locations(outputs["cls"], strides=[8, 16, 32])
    locations  = locations.to(cls_preds.device)

    B, N, C = cls_preds.shape
    _, P, Hp, Wp = proto.shape

    preds = []

    for i in range(B):
        scores_all, labels_all = torch.sigmoid(cls_preds[i]).max(dim=-1)  # [N]

        keep = scores_all > score_thresh
        if keep.sum() == 0:
            preds.append({
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,),   dtype=torch.float32),
                "labels": torch.zeros((0,),   dtype=torch.int64),
                "masks":  torch.zeros((0, img_size, img_size), dtype=torch.bool),
            })
            continue

        kept_scores = scores_all[keep]
        kept_labels = labels_all[keep]
        kept_locs   = locations[keep]          # [k, 2]
        kept_boxes  = box_preds[i][keep]       # [k, 4]  (ltrb normalized)
        kept_coefs  = coef_preds[i][keep]      # [k, P]

        # Decode ltrb → x1y1x2y2
        l = kept_locs[:, 0] - kept_boxes[:, 0] * img_size
        t = kept_locs[:, 1] - kept_boxes[:, 1] * img_size
        r = kept_locs[:, 0] + kept_boxes[:, 2] * img_size
        b = kept_locs[:, 1] + kept_boxes[:, 3] * img_size
        decoded_boxes = torch.stack([l, t, r, b], dim=1).clamp(0, img_size)

        # NMS per class
        from torchvision.ops import batched_nms
        nms_keep = batched_nms(decoded_boxes, kept_scores, kept_labels, iou_threshold=0.5)
        decoded_boxes = decoded_boxes[nms_keep]
        kept_scores   = kept_scores[nms_keep]
        kept_labels   = kept_labels[nms_keep]
        kept_coefs    = kept_coefs[nms_keep]

        # Decode masks
        mask_logits = kept_coefs @ proto[i].view(P, -1)          # [k, Hp*Wp]
        mask_logits = mask_logits.view(-1, Hp, Wp)
        mask_probs  = torch.sigmoid(mask_logits)
        masks_full  = F.interpolate(
            mask_probs.unsqueeze(1),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(1) > 0.5                                         # [k, H, W] bool

        preds.append({
            "boxes":  decoded_boxes,
            "scores": kept_scores,
            "labels": kept_labels,
            "masks":  masks_full,
        })

    return preds


# ─────────────────────────────────────────────────────────────
# Build GT list for torchmetrics
# ─────────────────────────────────────────────────────────────
def build_gt_list(targets, img_size=550):
    import torch.nn.functional as F
    gt_list = []
    for t in targets:
        masks = t["masks"]  # [M, H, W]
        if masks.shape[0] > 0 and masks.shape[-1] != img_size:
            masks = F.interpolate(
                masks.unsqueeze(1).float(),
                size=(img_size, img_size),
                mode="nearest"
            ).squeeze(1).bool()
        else:
            masks = masks.bool()

        gt_list.append({
            "boxes":  t["boxes"],
            "labels": t["labels"],
            "masks":  masks,
        })
    return gt_list


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────
def train(config):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = config.get("img_size", 550)
    print(f"Training on: {device}")

    # ── Data ────────────────────────────────────────────────
    train_loader, val_loader = get_sbd_dataloaders(
        root=config["data_root"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        img_size=img_size,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────
    model = MVP_Seg(
        model_name=config["backbone"],
        num_classes=config["num_classes"],
        num_prototypes=config["num_prototypes"],
    ).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = False
    print(f"Backbone frozen for {config['warmup_epochs']} warmup epochs")

    # ── Optimizer / Scheduler / Loss ────────────────────────
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = Model_Loss(num_classes=config["num_classes"])
    scaler    = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ── Resume ───────────────────────────────────────────────
    start_epoch = 0
    best_loss   = float("inf")
    if config.get("resume") and os.path.exists(config["resume_path"]):
        ckpt = torch.load(config["resume_path"], map_location=device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_loss   = ckpt["best_loss"]
            print(f"✅ Resumed from {config['resume_path']} at epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt)
            print(f"✅ Loaded pretrained weights from {config['resume_path']}")

    # ── Training loop ────────────────────────────────────────
    for epoch in range(start_epoch, config["epochs"]):

        # Unfreeze backbone after warmup
        if epoch == config["warmup_epochs"]:
            print(f"\nEpoch {epoch+1}: Unfreezing backbone")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = Adam([
                {"params": model.backbone.parameters(),  "lr": config["lr"] * 0.1},
                {"params": model.neck.parameters()},
                {"params": model.proto.parameters()},
                {"params": model.pred_head.parameters()},
            ], lr=config["lr"], weight_decay=config["weight_decay"])
            scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"] - epoch)

        # LR warmup (manual)
        if epoch < config["warmup_epochs"]:
            warmup_factor = (epoch + 1) / config["warmup_epochs"]
            for g in optimizer.param_groups:
                g["lr"] = config["lr"] * warmup_factor

        # ── Train step ──────────────────────────────────────
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            images, targets = batch

            if batch_idx == 0:
                if len(targets) > 0 and len(targets[0]["labels"]) > 0:
                    print(f"  Labels range: {targets[0]['labels'].min().item()} "
                          f"- {targets[0]['labels'].max().item()}")
                else:
                    print("  No objects in first batch")

            images  = images.to(device)
            targets = move_targets_to_device(targets, device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs   = model(images)
                loss_dict = criterion(outputs, targets)
                loss      = loss_dict["loss"]

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=10.0,
            )
            scaler.step(optimizer)
            scaler.update()

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
        scheduler.step()
        print(f"\nEpoch {epoch+1} | Train Loss: {avg_loss:.4f}")

        # ── Validation + AP ─────────────────────────────────
        model.eval()
        val_loss = 0.0

        # Tính cả box AP và mask AP
        metric = MeanAveragePrecision(
            iou_type=["bbox", "segm"],
            box_format="xyxy",
        ).to(device)

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                images, targets = batch
                images  = images.to(device)
                targets = move_targets_to_device(targets, device)

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    outputs   = model(images)
                    loss_dict = criterion(outputs, targets)

                val_loss += loss_dict["loss"].item()

                # Decode predictions và GT cho AP
                preds   = decode_outputs(outputs, score_thresh=0.3, img_size=img_size)
                gt_list = build_gt_list(targets, img_size=img_size)

                metric.update(preds, gt_list)

        val_loss /= max(len(val_loader), 1)

        # Compute AP metrics
        ap_results = metric.compute()
        map_bbox   = ap_results["map"].item()
        map_50_bbox = ap_results.get("map_50", torch.tensor(0.0)).item()
        map_segm   = ap_results.get("map_segm", torch.tensor(0.0)).item() \
                     if "map_segm" in ap_results else \
                     ap_results.get("map", torch.tensor(0.0)).item()

        print(f"Validation Loss : {val_loss:.4f}")
        print(f"  Box  mAP@0.5:0.95 = {map_bbox:.4f}")
        print(f"  Box  mAP@0.50     = {map_50_bbox:.4f}")
        print(f"  Mask mAP@0.5:0.95 = {map_segm:.4f}")

        # ── Save checkpoint ──────────────────────────────────
        os.makedirs(config["save_dir"], exist_ok=True)
        ckpt_data = {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "scaler_state":   scaler.state_dict(),
            "best_loss":      best_loss,
            "map_bbox":       map_bbox,
            "map_segm":       map_segm,
        }

        torch.save(ckpt_data, os.path.join(config["save_dir"], f"epoch_{epoch+1}.pth"))

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_data["best_loss"] = best_loss
            torch.save(ckpt_data, os.path.join(config["save_dir"], "best.pth"))
            print(f"🔥 Best model saved (val_loss={best_loss:.4f}, "
                  f"box_mAP={map_bbox:.4f}, mask_mAP={map_segm:.4f})")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = {
        "data_root":    "/content/SBD",
        "save_dir":     "checkpoints",

        "backbone":      "nvidia/MambaVision-T-1K",
        "num_classes":   20,
        "num_prototypes": 32,
        "img_size":      550,

        "batch_size":    2,
        "num_workers":   2,

        "lr":            1e-4,
        "weight_decay":  1e-4,

        "epochs":        10,
        "warmup_epochs": 3,

        "resume":        False,
        "resume_path":   "checkpoints/best.pth",
    }

    train(config)