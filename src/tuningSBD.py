import os
import torch
from torch.optim import AdamW
import numpy as np

from src.models.mvp_seg import MVP_Seg
from src.dataset.sbd_dataset import get_sbd_dataloaders
from src.models.loss import Model_Loss
from src.eval_sbd import evaluate_sbd


def move_targets_to_device(targets, device):
    new_targets = []
    for t in targets:
        new_t = {}
        for k, v in t.items():
            new_t[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        new_targets.append(new_t)
    return new_targets


# ─────────────────────────────────────────
# OPTIMIZER BUILDER
# ─────────────────────────────────────────
def build_optimizer(model, base_lr, weight_decay, backbone_lr_ratio=0.1):
    """
    Finetune: backbone dùng lr rất nhỏ (ratio 0.1) để giữ đặc trưng đã học từ COCO,
    tránh catastrophic forgetting. Head / Neck học với base_lr đầy đủ.
    """
    return AdamW([
        {"params": model.backbone.parameters(),  "lr": base_lr * backbone_lr_ratio},
        {"params": model.neck.parameters(),       "lr": base_lr},
        {"params": model.proto.parameters(),      "lr": base_lr},
        {"params": model.pred_head.parameters(),  "lr": base_lr},
    ], weight_decay=weight_decay)


# ─────────────────────────────────────────
# POLY LR
# ─────────────────────────────────────────
def poly_lr_scheduler(optimizer, base_lrs, curr_iter, max_iter, power=1.0):
    factor = (1 - curr_iter / max_iter) ** power
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = base_lrs[i] * factor


# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────
def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # ── DATA ────────────────────────────────────────────────
    train_loader, val_loader = get_sbd_dataloaders(
        root        = config["data_root"],
        batch_size  = config["batch_size"],
        num_workers = config["num_workers"],
        img_size    = config.get("img_size", 550),
        verbose     = True,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── MODEL ────────────────────────────────────────────────
    model = MVP_Seg(
        model_name     = config["backbone"],
        pretrained     = False,   # Không load HuggingFace pretrained — ta load COCO ckpt bên dưới
        num_classes    = config["num_classes"],
        num_prototypes = config["num_prototypes"],
    ).to(device)

    # ── LOAD COCO PRETRAINED ─────────────────────────────────
    # Load Backbone + Neck + ProtoNet từ COCO checkpoint.
    # cls_head bị bỏ qua tự động vì shape 80-class != 20-class,
    # và prior_prob init trong Prediction_Head.__init__ đã reset nó đúng cách.
    finetune_path = config.get("finetune_path", "")
    if os.path.exists(finetune_path):
        print(f"\n[Finetune] Loading COCO checkpoint: {finetune_path}")
        ckpt       = torch.load(finetune_path, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)

        model_dict      = model.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        skipped = [k for k in state_dict if k not in pretrained_dict]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"[Finetune] Loaded  : {len(pretrained_dict)} tensors")
        print(f"[Finetune] Skipped : {len(skipped)} tensors")
        if skipped:
            print(f"           → {skipped[:10]}{'...' if len(skipped) > 10 else ''}")
    else:
        print("[WARNING] finetune_path không tồn tại, train từ đầu!")

    # ── OPTIMIZER & LOSS ─────────────────────────────────────
    optimizer = build_optimizer(model, base_lr=config["lr"], weight_decay=config["weight_decay"])
    base_lrs  = [g["lr"] for g in optimizer.param_groups]

    criterion = Model_Loss(num_classes=config["num_classes"], alpha_box=1.5)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    total_iters  = config["epochs"] * len(train_loader)
    warmup_iters = 500   # Finetune → warmup ngắn hơn COCO

    start_epoch = 0
    best_AP50   = 0.0

    # ── RESUME ───────────────────────────────────────────────
    if config.get("resume", False):
        resume_path = config.get("resume_path", "")
        if os.path.exists(resume_path):
            print(f"[Resume] Loading: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scaler_state" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_AP50   = ckpt.get("best_AP50", 0.0)
            print(f"[Resume] Continuing from epoch {start_epoch}, best AP50: {best_AP50:.4f}")

    # ── TRAIN LOOP ───────────────────────────────────────────
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_loss = total_cls = total_box = total_msk = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            images, targets = batch
            images  = images.to(device)
            targets = move_targets_to_device(targets, device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs   = model(images)
                loss_dict = criterion(outputs, targets)

            loss = loss_dict["loss"]
            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=10.0,
            )

            scaler.step(optimizer)
            scaler.update()

            # ── LR Scheduling ──────────────────────────────
            global_iter = epoch * len(train_loader) + batch_idx
            if global_iter < warmup_iters:
                factor = (global_iter + 1) / warmup_iters
                for i, g in enumerate(optimizer.param_groups):
                    g["lr"] = base_lrs[i] * factor
            else:
                poly_lr_scheduler(optimizer, base_lrs, global_iter, total_iters, power=1.0)

            total_loss += loss.item()
            total_cls  += loss_dict["loss_cls"].item()
            total_box  += loss_dict["loss_box"].item()
            total_msk  += loss_dict["loss_mask"].item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{config['epochs']}] "
                    f"Step [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(cls: {loss_dict['loss_cls'].item():.4f} "
                    f"box: {loss_dict['loss_box'].item():.3f} "
                    f"mask: {loss_dict['loss_mask'].item():.3f})"
                )

        n = max(len(train_loader), 1)
        print(
            f"\nEpoch {epoch+1} | "
            f"Loss: {total_loss/n:.4f}  "
            f"cls: {total_cls/n:.4f}  "
            f"box: {total_box/n:.4f}  "
            f"mask: {total_msk/n:.4f}"
        )

        # ── CHECKPOINT BASE ──────────────────────────────────
        os.makedirs(config["save_dir"], exist_ok=True)

        checkpoint = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "best_AP50":       best_AP50,
        }

        # Ghi đè last.pth mỗi epoch để resume an toàn khi Colab disconnect
        torch.save(checkpoint, os.path.join(config["save_dir"], "last.pth"))

        # Epoch checkpoint mỗi 10 epoch
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(config["save_dir"], f"epoch_{epoch+1}.pth"))
            print(f"[Save] epoch_{epoch+1}.pth")

        # ── VALIDATION AP50 mỗi eval_every epoch ─────────────
        # Dùng evaluate_sbd() từ eval_sbd.py — đúng format cho SBD val set
        # Chỉ lấy AP50 để quyết định best: nhanh và đủ ý nghĩa cho giai đoạn finetune
        if (epoch + 1) % config.get("eval_every", 5) == 0:
            print(f"\n[Eval] Đánh giá AP50 trên SBD val (Epoch {epoch+1})...")
            results = evaluate_sbd(
                model_path   = os.path.join(config["save_dir"], "last.pth"),
                data_root    = config["data_root"],
                num_classes  = config["num_classes"],
                score_thresh = 0.05,
                device       = device,
                verbose      = True,
            )
            current_AP50 = results["AP50"]
            print(f"[Eval] AP50: {current_AP50*100:.2f}%  (Best: {best_AP50*100:.2f}%)")

            if current_AP50 > best_AP50:
                best_AP50 = current_AP50
                checkpoint["best_AP50"] = best_AP50
                torch.save(checkpoint, os.path.join(config["save_dir"], "best.pth"))
                print(f"[Save] best.pth  (AP50: {best_AP50*100:.2f}%)")


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
if __name__ == "__main__":
    config = {
        # ── Paths ─────────────────────────────────────────────
        "data_root":     "/content/SBD",
        "finetune_path": "/content/drive/MyDrive/checkpoints/coco/epoch_50.pth",
        "save_dir":      "/content/drive/MyDrive/checkpoints/sbd",

        # ── Model ─────────────────────────────────────────────
        "backbone":       "nvidia/MambaVision-T-1K",
        "num_classes":    20,
        "num_prototypes": 32,

        # ── Training ──────────────────────────────────────────
        "img_size":      550,
        "batch_size":    8,
        "num_workers":   2,

        "lr":            5e-5,
        "weight_decay":  0.01,

        "epochs":        60,
        "eval_every":    10,    # Chạy evaluate_sbd() mỗi 10 epoch

        # ── Resume ────────────────────────────────────────────
        "resume":        False,
        "resume_path":   "/content/drive/MyDrive/checkpoints/sbd/last.pth",
    }

    train(config)