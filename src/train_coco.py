import os
import torch
from torch.optim import AdamW

from src.models.mvp_seg import MVP_Seg
from src.models.loss import Model_Loss
from src.dataset.coco_dataset import get_coco_dataloaders


def move_targets_to_device(targets, device):
    moved = []
    for t in targets:
        moved.append({
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in t.items()
        })
    return moved


def build_optimizer(model, base_lr, weight_decay, backbone_lr_ratio=0.5):
    return AdamW([
        {"params": model.backbone.parameters(), "lr": base_lr * backbone_lr_ratio},
        {"params": model.neck.parameters(), "lr": base_lr},
        {"params": model.proto.parameters(), "lr": base_lr},
        {"params": model.pred_head.parameters(), "lr": base_lr},
    ], weight_decay=weight_decay)


def poly_lr_scheduler(optimizer, base_lrs, curr_iter, max_iter, power=1.0):
    factor = (1 - curr_iter / max_iter) ** power
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = base_lrs[i] * factor


def _build_checkpoint(epoch, model, optimizer, scaler, best_val_loss):
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }


def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_loader, val_loader = get_coco_dataloaders(
        data_root=config["data_root"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        subset_size=config.get("subset_size", 10000),
        img_size=config["img_size"],
        val_subset_size=config.get("val_subset_size", None),
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = MVP_Seg(
        model_name=config["backbone"],
        pretrained=config.get("pretrained", True),
        num_classes=config["num_classes"],
        num_prototypes=config["num_prototypes"],
    ).to(device)

    if config.get("freeze_backbone", False):
        for p in model.backbone.parameters():
            p.requires_grad = False
        model.backbone.eval()

    optimizer = build_optimizer(
        model=model,
        base_lr=config["lr"],
        weight_decay=config["weight_decay"],
        backbone_lr_ratio=config.get("backbone_lr_ratio", 0.5),
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    criterion = Model_Loss(
        num_classes=config["num_classes"],
        strides=config.get("strides", [8, 16, 32]),
        img_size=config["img_size"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    total_iters = config["epochs"] * max(len(train_loader), 1)
    warmup_iters = config.get("warmup_iters", 1500)
    grad_clip = config.get("grad_clip", 10.0)

    start_epoch = 0
    best_val_loss = float("inf")

    resume_path = config.get("resume_path", "")
    if config.get("resume", False) and resume_path and os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            for g in optimizer.param_groups:
                g["weight_decay"] = config["weight_decay"]

        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])

        start_epoch = ckpt.get("epoch", 0) + 1
        # backward compatible with older checkpoints using "best_loss"
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("best_loss", float("inf")))
        print(f"Resume epoch: {start_epoch}")

    os.makedirs(config["save_dir"], exist_ok=True)

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_train_loss = 0.0
        seen_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            images, targets = batch
            images = images.to(device, non_blocking=True)
            targets = move_targets_to_device(targets, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()

            global_iter = epoch * len(train_loader) + batch_idx
            if warmup_iters > 0 and global_iter < warmup_iters:
                warmup_factor = global_iter / warmup_iters
                for i, g in enumerate(optimizer.param_groups):
                    g["lr"] = base_lrs[i] * warmup_factor
            else:
                poly_lr_scheduler(optimizer, base_lrs, global_iter, total_iters, power=1.0)

            total_train_loss += float(loss.item())
            seen_batches += 1

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{config['epochs']}] Step [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(cls: {loss_dict['loss_cls'].item():.4f}, "
                    f"box: {loss_dict['loss_box'].item():.4f}, "
                    f"mask: {loss_dict['loss_mask'].item():.4f})"
                )

        avg_train_loss = total_train_loss / max(seen_batches, 1)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images, targets = batch
                images = images.to(device, non_blocking=True)
                targets = move_targets_to_device(targets, device)

                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                total_val_loss += float(loss_dict["loss"].item())
                val_seen += 1

        avg_val_loss = total_val_loss / max(val_seen, 1)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

        checkpoint = _build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            best_val_loss=best_val_loss,
        )

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(config["save_dir"], f"epoch_{epoch+1}.pth"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, os.path.join(config["save_dir"], "best.pth"))
            print(f"Best checkpoint saved: {best_val_loss:.4f}")


if __name__ == "__main__":
    config = {
        "data_root": "/content/coco",
        "save_dir": "checkpoints_coco",
        "backbone": "nvidia/MambaVision-T-1K",
        "pretrained": True,
        "num_classes": 80,
        "num_prototypes": 32,
        "img_size": 550,
        "strides": [8, 16, 32],
        "subset_size": 10000,
        "val_subset_size": None,
        "batch_size": 5,
        "num_workers": 2,
        "lr": 1e-5,
        "weight_decay": 0.01,
        "backbone_lr_ratio": 0.5,
        "epochs": 30,
        "warmup_iters": 1500,
        "grad_clip": 10.0,
        "freeze_backbone": False,
        "resume": False,
        "resume_path": "checkpoints_coco/best.pth",
    }
    train(config)
