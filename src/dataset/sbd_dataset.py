import os
import torch
import numpy as np
import tarfile
import urllib.request
import shutil
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import SBDataset
from scipy import ndimage
import albumentations as A
from albumentations.pytorch import ToTensorV2
 
 
# ─────────────────────────────────────────────
# 1. SBInstanceDataset
# ─────────────────────────────────────────────
class SBInstanceDataset(Dataset):
    def __init__(self, data_root: str, image_set: str = "train", img_size: int = 550):
        self.dataset = SBDataset(
            root=data_root,
            image_set=image_set,
            mode="segmentation",
            download=False
        )
        self.img_size = img_size
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
 
    def __len__(self):
        return len(self.dataset)
 
    def __getitem__(self, idx: int):
        image, mask = self.dataset[idx]
        image = np.array(image)   # H × W × 3  uint8
        mask  = np.array(mask)    # H × W      uint8
 
        # ── collect per-instance masks & labels ──────────────────────────
        class_ids = np.unique(mask)
        class_ids = class_ids[(class_ids != 0) & (class_ids != 255)]
 
        masks_list, labels_list = [], []
        for cls_id in class_ids:
            binary = (mask == cls_id).astype(np.uint8)
            labeled_array, num_instances = ndimage.label(binary)
 
            for inst_idx in range(1, num_instances + 1):
                inst_mask = (labeled_array == inst_idx).astype(np.uint8)
                if inst_mask.sum() < 20:
                    continue
                masks_list.append(inst_mask)
                labels_list.append(int(cls_id) - 1)  # 1-20 → 0-19
 
        # ── transform (resize + normalize) ───────────────────────────────
        if len(masks_list) > 0:
            out   = self.transform(image=image, masks=masks_list)
            image = out["image"]          # [C, H, W] float tensor
            masks_list = out["masks"]     # list of H×W numpy arrays
        else:
            out   = self.transform(image=image, masks=[])
            image = out["image"]
 
        h, w = image.shape[1], image.shape[2]
 
        # ── build valid boxes / labels / masks ───────────────────────────
        valid_boxes, valid_labels, valid_masks = [], [], []
 
        for m, lbl in zip(masks_list, labels_list):
            m_np = np.array(m)
            ys, xs = np.where(m_np)
            if len(xs) == 0:
                continue
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            if x2 <= x1 or y2 <= y1:
                continue
            valid_boxes.append([x1, y1, x2, y2])
            valid_labels.append(lbl)
            valid_masks.append(torch.tensor(m_np, dtype=torch.float32))
 
        if len(valid_boxes) == 0:
            return {
                "image":  image,
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,),   dtype=torch.int64),
                "masks":  torch.zeros((0, h, w), dtype=torch.float32),
                "img_id": idx,
            }
 
        return {
            "image":  image,
            "boxes":  torch.tensor(valid_boxes, dtype=torch.float32),
            "labels": torch.tensor(valid_labels, dtype=torch.int64),
            "masks":  torch.stack(valid_masks),
            "img_id": idx,
        }
 
 
# ─────────────────────────────────────────────
# 2. collate_fn
# ─────────────────────────────────────────────
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
 
    images  = torch.stack([b["image"] for b in batch])
    targets = [
        {
            "boxes":  b["boxes"],
            "labels": b["labels"],
            "masks":  b["masks"],
            "img_id": b["img_id"],
        }
        for b in batch
    ]
    return images, targets
 
 
# ─────────────────────────────────────────────
# 3. Public API
# ─────────────────────────────────────────────
def get_sbd_dataloaders(
    root: str       = "/data/SBD",
    batch_size: int = 4,
    num_workers: int = 2,
    img_size: int   = 550,
    verbose: bool   = True,
):
    os.makedirs(root, exist_ok=True)
 
    dataset_path = os.path.join(root, "dataset")
    os.makedirs(dataset_path, exist_ok=True)
 
    # Fix folder structure if needed
    for item in ["img", "cls", "inst", "train.txt", "val.txt"]:
        src = os.path.join(root, item)
        dst = os.path.join(dataset_path, item)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
 
    # Download if not exists
    if not os.path.exists(os.path.join(dataset_path, "train.txt")):
        url = (
            "https://www2.eecs.berkeley.edu/Research/Projects/CS/"
            "vision/grouping/semantic_contours/benchmark.tgz"
        )
        tgz_path = os.path.join(root, "benchmark.tgz")
 
        if verbose:
            print("[INFO] Downloading SBD dataset...")
        urllib.request.urlretrieve(url, tgz_path)
 
        if verbose:
            print("[INFO] Extracting...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(root)
 
        src_release = os.path.join(root, "benchmark_RELEASE", "dataset")
        for item in os.listdir(src_release):
            shutil.move(os.path.join(src_release, item), dataset_path)
 
        shutil.rmtree(os.path.join(root, "benchmark_RELEASE"))
 
    # ── datasets ─────────────────────────────────────────────────────────
    train_dataset = SBInstanceDataset(
        data_root=dataset_path,
        image_set="train",
        img_size=img_size,
    )
    val_dataset = SBInstanceDataset(
        data_root=dataset_path,
        image_set="val",
        img_size=img_size,
    )
 
    # ── dataloaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
 
    if verbose:
        print(f"[INFO] Train: {len(train_dataset)} samples")
        print(f"[INFO] Val:   {len(val_dataset)} samples")
 
    return train_loader, val_loader