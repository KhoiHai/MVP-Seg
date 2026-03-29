import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.mvp_seg import MVPSeg
from src.dataset import get_dataloaders
from src.loss import YOLACTLoss


def train(config):
    '''
    Full training loop for MVP-Seg
        - Backbone is frozen during warm-up epochs to stabilize early training
        - After warm-up, backbone is unfrozen with a lower learning rate (10x smaller)
        - Checkpoints are saved every epoch for Colab resume support
        - Best model (lowest train loss) is separately saved as best.pth
    Args:
        config: dict containing all hyperparameters (see __main__ block below)
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Build DataLoaders
    train_loader, val_loader = get_dataloaders(
        data_root    = config["data_root"],
        batch_size   = config["batch_size"],
        num_workers  = config["num_workers"],
        subset_size  = config["subset_size"]
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Build model
    model = MVPSeg(
        model_name     = config["backbone"],
        num_classes    = config["num_classes"],
        num_prototypes = config["num_prototypes"]
    ).to(device)

    # Freeze backbone during warm-up to let the new heads stabilize first
    for param in model.backbone.parameters():
        param.requires_grad = False
    print(f"Backbone frozen for {config['warmup_epochs']} warm-up epochs")

    # Optimizer only covers unfrozen params initially
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config["lr"],
        betas        = (0.9, 0.999),
        weight_decay = config["weight_decay"]
    )

    # Cosine annealing decays LR smoothly to near 0 by the final epoch
    scheduler = CosineAnnealingLR(optimizer, T_max = config["epochs"])

    criterion = YOLACTLoss(num_classes = config["num_classes"])

    # Resume from checkpoint if provided (useful when Colab session is interrupted)
    start_epoch = 0
    best_loss   = float("inf")
    if config.get("resume") and os.path.exists(config["resume"]):
        ckpt = torch.load(config["resume"], map_location = device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt.get("loss", float("inf"))
        print(f"Resumed from checkpoint: epoch {start_epoch}, loss {best_loss:.4f}")

    os.makedirs(config["save_dir"], exist_ok = True)

    # Training loop
    for epoch in range(start_epoch, config["epochs"]):

        # Unfreeze backbone after warm-up with a reduced learning rate
        if epoch == config["warmup_epochs"]:
            print(f"\nEpoch {epoch + 1}: Unfreezing backbone with lr = {config['lr'] * 0.1:.6f}")
            for param in model.backbone.parameters():
                param.requires_grad = True

            # Rebuild optimizer with separate LR groups
            optimizer = Adam([
                {"params": model.backbone.parameters(),  "lr": config["lr"] * 0.1},
                {"params": model.neck.parameters()},
                {"params": model.protonet.parameters()},
                {"params": model.pred_head.parameters()},
            ], lr = config["lr"], weight_decay = config["weight_decay"])

        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            images, targets = batch
            images = images.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs   = model(images)
            loss_dict = criterion(outputs, targets)
            loss      = loss_dict["loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10.0)

            optimizer.step()

            total_loss += loss.item()

            # Print progress every 50 steps
            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{config['epochs']}] "
                    f"Step [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(cls: {loss_dict['loss_cls'].item():.3f}  "
                    f"box: {loss_dict['loss_box'].item():.3f}  "
                    f"mask: {loss_dict['loss_mask'].item():.3f})"
                )

        avg_loss = total_loss / max(len(train_loader), 1)
        scheduler.step()

        print(
            f"\nEpoch {epoch + 1} summary | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save epoch checkpoint (allows resuming from any epoch on Colab)
        ckpt_path = os.path.join(config["save_dir"], f"epoch_{epoch + 1}.pth")
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss":      avg_loss
        }, ckpt_path)

        # Overwrite best.pth whenever we achieve a new lowest loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                os.path.join(config["save_dir"], "best.pth")
            )
            print(f"Best model saved (loss: {best_loss:.4f})")


# ------------------------------
# TRAINING CONFIGURATION
# ------------------------------
if __name__ == "__main__":
    config = {
        # Paths
        "data_root":       "data/coco",
        "save_dir":        "checkpoints",

        # Model
        "backbone":        "nvidia/MambaVision-T-1K",
        "num_classes":     80,
        "num_prototypes":  32,

        # Data
        "batch_size":      5,          # Per proposal
        "num_workers":     2,
        "subset_size":     10000,      # COCO subset ~10k train images

        # Optimizer (per proposal)
        "lr":              1e-4,
        "weight_decay":    1e-4,

        # Scheduler
        "epochs":          200,
        "warmup_epochs":   5,          # Backbone frozen for first 5 epochs

        # Resume (set to checkpoint path to continue interrupted training)
        # Example: "checkpoints/epoch_20.pth"
        "resume":          None,
    }

    train(config)