import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.mvp_seg import MVP_Seg
from src.dataset.coco_dataset import get_coco_dataloaders
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
        data_root=config["data_root"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        subset_size=config["subset_size"]
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
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        betas=(0.9, 0.999),
        weight_decay=config["weight_decay"]
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    criterion = Model_Loss(num_classes=config["num_classes"])

    # AMP (mixed precision)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # -------------------------
    # RESUME
    # -------------------------
    start_epoch = 0
    best_loss = float("inf")

    if config.get("resume") and os.path.exists(config["resume"]):
        ckpt = torch.load(config["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, loss {best_loss:.4f}")

    os.makedirs(config["save_dir"], exist_ok=True)

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    for epoch in range(start_epoch, config["epochs"]):

        # -------------------------
        # UNFREEZE BACKBONE
        # -------------------------
        if epoch == config["warmup_epochs"]:
            print(f"\nEpoch {epoch+1}: Unfreezing backbone")

            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer = Adam([
                {"params": model.backbone.parameters(), "lr": config["lr"] * 0.1},
                {"params": model.neck.parameters()},
                {"params": model.proto.parameters()},
                {"params": model.pred_head.parameters()},
            ], lr=config["lr"], weight_decay=config["weight_decay"])

            # FIX: scheduler reset đúng
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config["epochs"] - epoch
            )

        model.train()
        total_loss = 0.0

        # -------------------------
        # LR WARMUP (REAL)
        # -------------------------
        if epoch < config["warmup_epochs"]:
            warmup_factor = (epoch + 1) / config["warmup_epochs"]
            for g in optimizer.param_groups:
                g["lr"] = config["lr"] * warmup_factor

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            images, targets = batch
            images = images.to(device)
            targets = move_targets_to_device(targets, device)

            optimizer.zero_grad()

            # -------------------------
            # FORWARD (AMP)
            # -------------------------
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["loss"]

            # -------------------------
            # BACKWARD
            # -------------------------
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

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch+1} summary | "
            f"Train Loss: {avg_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )

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
        ckpt_path = os.path.join(config["save_dir"], f"epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss
        }, ckpt_path)

        # Save best (based on val)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(config["save_dir"], "best.pth")
            )
            print(f"Best model saved (val loss: {best_loss:.4f})")


# ------------------------------
# CONFIG
# ------------------------------
if __name__ == "__main__":
    config = {
        "data_root": "data/coco",
        "save_dir": "checkpoints",

        "backbone": "nvidia/MambaVision-T-1K",
        "num_classes": 80,
        "num_prototypes": 32,

        "batch_size": 4,       
        "num_workers": 2,
        "subset_size": 100,   

        "lr": 1e-4,
        "weight_decay": 1e-4,

        "epochs": 30,
        "warmup_epochs": 3,

        "resume": None,
    }

    train(config)