import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import SBDataset
import numpy as np
from tqdm import tqdm  # Thêm progress bar

class SBInstanceDataset(Dataset):
    """
    SBDataset loader for instance segmentation
    Converts SBD semantic masks to instance masks
    """
    def __init__(self, root, image_set="train", transforms=None, verbose=True):
        """
        Args:
            root: folder to download/extract SBD dataset
            image_set: "train" / "val" / "train_noval"
            transforms: torchvision or albumentations transforms
            verbose: show progress when loading individual images
        """
        print(f"[INFO] Initializing SBDataset for '{image_set}'...")
        start_time = time.time()
        self.dataset = SBDataset(
            root=root,
            image_set=image_set,
            mode="segmentation",
            download=True
        )
        self.transforms = transforms
        self.verbose = verbose
        if self.verbose:
            print(f"[INFO] {len(self.dataset)} images loaded in {time.time() - start_time:.2f}s.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.verbose and idx % 50 == 0:
            print(f"[INFO] Loading image {idx}/{len(self.dataset)}")

        image, mask = self.dataset[idx]  # PIL.Image
        image = np.array(image)
        mask  = np.array(mask)

        # Convert mask to instances
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 255]  # ignore

        boxes, labels, masks_list = [], [], []

        for inst_id in instance_ids:
            m = (mask == inst_id).astype(np.uint8)
            ys, xs = np.where(m)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            boxes.append([x1, y1, x2, y2])
            labels.append(inst_id)
            masks_list.append(torch.tensor(m, dtype=torch.float32))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.float32)
        else:
            boxes  = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks  = torch.stack(masks_list, dim=0)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "img_id": idx
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    images  = torch.stack([b["image"] for b in batch])
    targets = [{
        "boxes":  b["boxes"],
        "labels": b["labels"],
        "masks":  b["masks"],
        "img_id": b["img_id"]
    } for b in batch]
    return images, targets

def get_sbd_dataloaders(root="datasets/SBD", batch_size=4, num_workers=2, img_size=550, verbose=True, val_split=0.1):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import random
    from tqdm import tqdm

    # -------------------------
    # Transforms
    # -------------------------
    if verbose:
        print("[TRANSFORM] Initialize transformation")
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.2,0.2,0.2, p=0.5),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])

    # -------------------------
    # Tách train/val từ train_noval
    # -------------------------
    train_noval_file = os.path.join(root, "train_noval.txt")
    train_file = os.path.join(root, "train.txt")
    val_file   = os.path.join(root, "val.txt")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        with open(train_noval_file, "r") as f:
            imgs = f.read().splitlines()
        random.shuffle(imgs)
        split_idx = int(len(imgs)*(1-val_split))
        train_imgs = imgs[:split_idx]
        val_imgs   = imgs[split_idx:]
        with open(train_file, "w") as f:
            f.write("\n".join(train_imgs))
        with open(val_file, "w") as f:
            f.write("\n".join(val_imgs))
        if verbose:
            print(f"[INFO] Created train.txt ({len(train_imgs)} images) and val.txt ({len(val_imgs)} images)")

    # -------------------------
    # Load SBInstanceDataset
    # -------------------------
    if verbose:
        print("[INFO] Creating SBInstanceDataset objects...")
    train_dataset = SBInstanceDataset(root=root, image_set="train", transforms=train_transform, verbose=False)

    print("[INFO] Loading all train images with progress:")
    for i in tqdm(range(len(train_dataset))):
        _ = train_dataset[i]  # chỉ để chạy và thấy tiến trình

    val_dataset   = SBInstanceDataset(root=root, image_set="val", transforms=val_transform, verbose=False)

    # -------------------------
    # DataLoaders
    # -------------------------
    if verbose:
        print("[INFO] Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # -------------------------
    # Progress log cho train loader
    # -------------------------
    if verbose:
        print("[INFO] Iterating through train_loader to show progress...")
        for i, (images, targets) in enumerate(tqdm(train_loader, desc="Loading train batches")):
            if i >= 5:  # chỉ hiển thị 5 batch đầu để demo
                break

    if verbose:
        print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    return train_loader, val_loader