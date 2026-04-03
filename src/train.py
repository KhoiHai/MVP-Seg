import os
import torch
from torch.optim import AdamW

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

# -------------------------
# OPTIMIZER BUILDER
# -------------------------
def build_optimizer(model, base_lr, weight_decay, backbone_lr_ratio=0.1):
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
# TRAIN
# -------------------------
def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # -------------------------
    # DATA
    # -------------------------
    train_loader, val_loader = get_sbd_dataloaders(
        root=config["data_root"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
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

    criterion = Model_Loss(num_classes=config["num_classes"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    total_iters = config["epochs"] * len(train_loader)
    warmup_iters = 1500

    start_epoch = 0
    best_loss = float("inf")

    # -------------------------
    # RESUME
    # -------------------------
    if config.get("resume", False):
        path = config.get("resume_path", "")
        if os.path.exists(path):
            print(f"🔄 Resuming from {path}")

            checkpoint = torch.load(path, map_location=device)

            model.load_state_dict(checkpoint["model_state"])

            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])

            if "scaler_state" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state"])

            start_epoch = checkpoint.get("epoch", 0) + 1
            best_loss = checkpoint.get("best_loss", float("inf"))

            print(f"✅ Resume at epoch {start_epoch}")

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
        # VALIDATION
        # -------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                images, targets = batch
                images = images.to(device)
                targets = move_targets_to_device(targets, device)

                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                val_loss += loss_dict["loss"].item()

        val_loss /= max(len(val_loader), 1)
        print(f"Validation Loss: {val_loss:.4f}")

        # -------------------------
        # SAVE
        # -------------------------
        os.makedirs(config["save_dir"], exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_loss": best_loss
        }

        torch.save(checkpoint, os.path.join(config["save_dir"], f"epoch_{epoch+1}.pth"))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, os.path.join(config["save_dir"], "best.pth"))
            print(f"🔥 Best model saved: {best_loss:.4f}")

# ------------------------------
# CONFIG
# ------------------------------
if __name__ == "__main__":
    config = {
        "data_root": "/content/SBD",
        "save_dir": "checkpoints",

        "backbone": "nvidia/MambaVision-T-1K",
        "num_classes": 20,
        "num_prototypes": 32,

        "batch_size": 2,
        "num_workers": 2,

        "lr": 1e-5,
        "weight_decay": 0.01,

        "epochs": 10,
        "warmup_epochs": 1,

        "resume": False,
        "resume_path": "checkpoints/best.pth",
    }

    train(config)