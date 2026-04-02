import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

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


def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # -------------------------
    # DATA
    # -------------------------
    train_loader, val_loader = get_sbd_dataloaders(
        root        = config["data_root"],
        batch_size  = config["batch_size"],
        num_workers = config["num_workers"],
        save_pt_dir = config["processed_dir"],
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # -------------------------
    # MODEL
    # -------------------------
    model = MVP_Seg(
        model_name     = config["backbone"],
        num_classes    = config["num_classes"],
        num_prototypes = config["num_prototypes"]
    ).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = False

    print(f"Backbone frozen for {config['warmup_epochs']} epochs")

    # -------------------------
    # OPTIMIZER
    # -------------------------
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config["lr"],
        weight_decay = config["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = Model_Loss(num_classes=config["num_classes"])
    scaler    = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # ── helper: unfreeze backbone & rebuild optimizer/scheduler ──────────
    def unfreeze_backbone(current_epoch):
        print(f"\nEpoch {current_epoch + 1}: Unfreezing backbone")
        for param in model.backbone.parameters():
            param.requires_grad = True

        new_optimizer = Adam([
            {"params": model.backbone.parameters(),  "lr": config["lr"] * 0.1},
            {"params": model.neck.parameters()},
            {"params": model.proto.parameters()},
            {"params": model.pred_head.parameters()},
        ], lr=config["lr"], weight_decay=config["weight_decay"])

        new_scheduler = CosineAnnealingLR(
            new_optimizer,
            T_max=config["epochs"] - current_epoch
        )
        return new_optimizer, new_scheduler

    # -------------------------
    # RESUME CHECKPOINT
    # -------------------------
    start_epoch = 0
    best_loss   = float("inf")

    if config.get("resume") and os.path.exists(config["resume_path"]):
        checkpoint = torch.load(config["resume_path"], map_location=device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss   = checkpoint["best_loss"]
            print(f"✅ Resumed from {config['resume_path']} at epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded pretrained model from {config['resume_path']}")

    # Nếu resume sau warmup → unfreeze ngay
    if start_epoch > config["warmup_epochs"]:
        print("Resuming after warmup → Unfreezing backbone")
        optimizer, scheduler = unfreeze_backbone(start_epoch)

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    for epoch in range(start_epoch, config["epochs"]):

        # Unfreeze đúng 1 lần tại epoch warmup_epochs
        if epoch == config["warmup_epochs"] and start_epoch <= config["warmup_epochs"]:
            optimizer, scheduler = unfreeze_backbone(epoch)

        model.train()
        total_loss = 0.0

        # LR warmup
        if epoch < config["warmup_epochs"]:
            warmup_factor = (epoch + 1) / config["warmup_epochs"]
            for g in optimizer.param_groups:
                g["lr"] = config["lr"] * warmup_factor

        # -------------------------
        # TRAIN STEP
        # -------------------------
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
                loss      = loss_dict["loss"]

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=10.0
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

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images, targets = batch
                images  = images.to(device)
                targets = move_targets_to_device(targets, device)

                outputs   = model(images)
                loss_dict = criterion(outputs, targets)
                val_loss += loss_dict["loss"].item()

        val_loss /= max(len(val_loader), 1)
        print(f"Validation Loss: {val_loss:.4f}")

        # -------------------------
        # SAVE
        # -------------------------
        os.makedirs(config["save_dir"], exist_ok=True)
        checkpoint_data = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "best_loss":       best_loss,
        }
        torch.save(checkpoint_data,
                   os.path.join(config["save_dir"], f"epoch_{epoch+1}.pth"))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint_data,
                       os.path.join(config["save_dir"], "best.pth"))
            print(f"🔥 Best model saved: {best_loss:.4f}")


# ------------------------------
# CONFIG
# ------------------------------
if __name__ == "__main__":
    config = {
        # ── Paths ─────────────────────────────────────────────────────────
        "data_root":     "/content/drive/MyDrive/MVP-Seg/data/SBD",
        "processed_dir": "/content/drive/MyDrive/MVP-Seg/processed_data",
        "save_dir":      "/content/drive/MyDrive/MVP-Seg/checkpoints",
        "resume_path":   "/content/drive/MyDrive/MVP-Seg/checkpoints/best.pth",

        # ── Model ─────────────────────────────────────────────────────────
        "backbone":       "nvidia/MambaVision-T-1K",
        "num_classes":    20,
        "num_prototypes": 32,

        # ── Training ──────────────────────────────────────────────────────
        "batch_size":    2,
        "num_workers":   2,
        "lr":            1e-4,
        "weight_decay":  1e-4,
        "epochs":        50,
        "warmup_epochs": 3,
        "resume":        False,
    }

    train(config)