import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import SBDataset
import numpy as np
import random
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -----------------------------
# Dataset convert semantic -> instance
# -----------------------------
class SBInstanceDataset(Dataset):
    """Convert SBD semantic masks to instance segmentation format."""
    def __init__(self, root, image_set="train", transforms=None, verbose=True):
        self.verbose = verbose
        self.dataset = SBDataset(root=root, image_set=image_set, mode="segmentation", download=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = np.array(image)
        mask  = np.array(mask)

        # Convert mask to instance masks
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 255]

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
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.float32)
        else:
            boxes  = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks  = torch.stack(masks_list, dim=0)

        # Transform
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image).permute(2,0,1).float()/255.0

        return {"image": image, "boxes": boxes, "labels": labels, "masks": masks, "img_id": idx}

# -----------------------------
# Collate function cho DataLoader
# -----------------------------
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch)==0:
        return None
    images = torch.stack([b["image"] for b in batch])
    targets = [{"boxes":b["boxes"], "labels":b["labels"], "masks":b["masks"], "img_id":b["img_id"]} for b in batch]
    return images, targets

# -----------------------------
# Loader chính
# -----------------------------
def get_sbd_dataloaders(root="data/SBD", batch_size=4, num_workers=2, img_size=550,
                        verbose=True, save_pt_dir="processed_data"):

    os.makedirs(save_pt_dir, exist_ok=True)

    # -------------------------
    # Check dataset folder
    # -------------------------
    img_folder = os.path.join(root, "img")
    cls_folder = os.path.join(root, "cls")
    if not (os.path.exists(img_folder) and os.path.exists(cls_folder)):
        if verbose:
            print(f"[INFO] Dataset not found in {root}. Downloading SBD dataset...")
        _ = SBDataset(root=root, image_set="train", mode="segmentation", download=True)
        _ = SBDataset(root=root, image_set="val", mode="segmentation", download=True)
        if verbose:
            print("[INFO] Download completed.")

    # -------------------------
    # Transforms
    # -------------------------
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
    # Load datasets
    # -------------------------
    train_dataset_full = SBInstanceDataset(root=root, image_set="train", transforms=train_transform, verbose=verbose)
    val_dataset_full   = SBInstanceDataset(root=root, image_set="val", transforms=val_transform, verbose=verbose)

    # -------------------------
    # Save to .pt for reuse
    # -------------------------
    train_pt_file = os.path.join(save_pt_dir, f"train_{img_size}.pt")
    val_pt_file   = os.path.join(save_pt_dir, f"val_{img_size}.pt")

    if os.path.exists(train_pt_file):
        if verbose: print(f"[INFO] Loading cached train dataset from {train_pt_file}")
        train_data = torch.load(train_pt_file)
    else:
        if verbose: print(f"[INFO] Saving train dataset to {train_pt_file} ...")
        train_data = [train_dataset_full[i] for i in tqdm(range(len(train_dataset_full)))]
        torch.save(train_data, train_pt_file)

    if os.path.exists(val_pt_file):
        if verbose: print(f"[INFO] Loading cached val dataset from {val_pt_file}")
        val_data = torch.load(val_pt_file)
    else:
        if verbose: print(f"[INFO] Saving val dataset to {val_pt_file} ...")
        val_data = [val_dataset_full[i] for i in tqdm(range(len(val_dataset_full)))]
        torch.save(val_data, val_pt_file)

    # -------------------------
    # Dataset wrapper cho DataLoader
    # -------------------------
    class PTDataset(Dataset):
        def __init__(self, data_list):
            self.data_list = data_list
        def __len__(self):
            return len(self.data_list)
        def __getitem__(self, idx):
            return self.data_list[idx]

    train_loader = DataLoader(PTDataset(train_data), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(PTDataset(val_data), batch_size=1,
                              shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    if verbose:
        print(f"[INFO] Done! Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader