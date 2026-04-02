import os
import torch
import numpy as np
import tarfile
import urllib.request
import shutil
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import SBDataset
from scipy import ndimage
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
# 1. SBInstanceDataset  –  lazy, one sample at a time
# ─────────────────────────────────────────────
class SBInstanceDataset(Dataset):
    """
    Wraps torchvision.SBDataset, converts semantic masks → per-instance
    binary masks, and applies a base transform (resize + normalise).

    Nothing is buffered in RAM beyond the current sample.
    """

    def __init__(self, data_root: str, image_set: str = "train", transforms=None):
        """
        Args:
            data_root  : Folder that contains train.txt / val.txt and img/ cls/ dirs.
                         (i.e. the inner  benchmark_RELEASE/dataset  directory)
            image_set  : "train" | "val"
            transforms : albumentations Compose pipeline (image + masks)
        """
        self.dataset    = SBDataset(root=data_root, image_set=image_set,
                                    mode="segmentation", download=False)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, mask = self.dataset[idx]
        image = np.array(image)           # H × W × 3  uint8
        mask  = np.array(mask)            # H × W      uint8  (semantic class ids)

        class_ids = np.unique(mask)
        class_ids = class_ids[(class_ids != 0) & (class_ids != 255)]

        masks_list, labels_list = [], []
        for cls_id in class_ids:
            binary         = (mask == cls_id).astype(np.uint8)
            labeled_array, num_instances = ndimage.label(binary)
            for inst_idx in range(1, num_instances + 1):
                inst_mask = (labeled_array == inst_idx).astype(np.uint8)
                if inst_mask.sum() < 20:          # skip tiny blobs
                    continue
                masks_list.append(inst_mask)
                labels_list.append(int(cls_id))

        if self.transforms and len(masks_list) > 0:
            transformed  = self.transforms(image=image, masks=masks_list)
            image        = transformed["image"]        # tensor C×H×W  float
            masks_list   = transformed["masks"]        # list of H×W arrays
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        h, w = image.shape[1], image.shape[2]
        valid_boxes, valid_labels, valid_masks = [], [], []
        for m, lbl in zip(masks_list, labels_list):
            m_np = np.array(m)
            ys, xs = np.where(m_np)
            if len(xs) == 0:
                continue
            valid_boxes.append([int(xs.min()), int(ys.min()),
                                 int(xs.max()), int(ys.max())])
            valid_labels.append(lbl)
            valid_masks.append(torch.tensor(m_np, dtype=torch.float32))

        if len(valid_boxes) == 0:
            return {
                "image":  image,
                "boxes":  torch.zeros((0, 4),           dtype=torch.float32),
                "labels": torch.zeros((0,),             dtype=torch.int64),
                "masks":  torch.zeros((0, h, w),        dtype=torch.float32),
                "img_id": idx,
            }

        return {
            "image":  image,
            "boxes":  torch.tensor(valid_boxes,  dtype=torch.float32),
            "labels": torch.tensor(valid_labels, dtype=torch.int64),
            "masks":  torch.stack(valid_masks),
            "img_id": idx,
        }

# ─────────────────────────────────────────────
# 2. Online-augmentation wrapper
# ─────────────────────────────────────────────
class AugmentedDataset(Dataset):
    """Applies random augmentations at training time (no extra RAM needed)."""

    def __init__(self, base_dataset: Dataset, aug, img_size: int):
        self.base     = base_dataset
        self.aug      = aug
        self.img_size = img_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]

        if self.aug is None or sample["masks"].shape[0] == 0:
            return sample

        img_np   = sample["image"].permute(1, 2, 0).numpy()
        masks_np = [sample["masks"][i].numpy()
                    for i in range(sample["masks"].shape[0])]

        aug_out = self.aug(image=img_np, masks=masks_np)

        valid_boxes, valid_labels, valid_masks = [], [], []
        for m_np, lbl in zip(aug_out["masks"], sample["labels"].tolist()):
            m_np = np.array(m_np)
            ys, xs = np.where(m_np)
            if len(xs) == 0:
                continue
            valid_boxes.append([int(xs.min()), int(ys.min()),
                                 int(xs.max()), int(ys.max())])
            valid_labels.append(lbl)
            valid_masks.append(torch.tensor(m_np, dtype=torch.float32))

        aug_img = torch.from_numpy(aug_out["image"]).permute(2, 0, 1)
        h, w    = aug_img.shape[1], aug_img.shape[2]

        return {
            "image":  aug_img,
            "boxes":  torch.tensor(valid_boxes,  dtype=torch.float32)
                      if valid_boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(valid_labels, dtype=torch.int64)
                      if valid_labels else torch.zeros((0,), dtype=torch.int64),
            "masks":  torch.stack(valid_masks)
                      if valid_masks else torch.zeros((0, h, w)),
            "img_id": sample["img_id"],
        }


# ─────────────────────────────────────────────
# 3. collate_fn
# ─────────────────────────────────────────────
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images  = torch.stack([b["image"] for b in batch])
    targets = [{"boxes":  b["boxes"],
                "labels": b["labels"],
                "masks":  b["masks"],
                "img_id": b["img_id"]} for b in batch]
    return images, targets


# ─────────────────────────────────────────────
# 4. Public API
# ─────────────────────────────────────────────
def get_sbd_dataloaders(
    root: str = "/data/SBD",
    batch_size: int = 4,
    num_workers: int = 2,
    img_size: int = 550,
    verbose: bool = True,
):
    os.makedirs(root, exist_ok=True)

    # ── chuẩn bị dataset path ─────────────────────────
    dataset_path = os.path.join(root, "dataset")
    os.makedirs(dataset_path, exist_ok=True)

    # fix cấu trúc folder nếu cần
    for item in ["img", "cls", "inst", "train.txt", "val.txt"]:
        src = os.path.join(root, item)
        dst = os.path.join(dataset_path, item)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
        else:
            print(f"[WARN] {dst} already exists, skipping move.")

    # ── download nếu chưa có ─────────────────────────
    if not os.path.exists(os.path.join(dataset_path, "train.txt")):
        url = ("https://www2.eecs.berkeley.edu/Research/Projects/CS/"
               "vision/grouping/semantic_contours/benchmark.tgz")

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

    # ── transform ─────────────────────────
    base_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # ── dataset ─────────────────────────
    train_dataset = SBInstanceDataset(
        data_root=dataset_path,
        image_set="train",
        transforms=base_transform
    )

    val_dataset = SBInstanceDataset(
        data_root=dataset_path,
        image_set="val",
        transforms=base_transform
    )

    # ── augmentation (chỉ train) ─────────────────────
    online_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2,
                      contrast=0.2,
                      saturation=0.2,
                      p=0.5),
    ])

    train_dataset = AugmentedDataset(train_dataset, online_aug, img_size)

    # ── dataloader ─────────────────────────
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

    if verbose:
        print(f"[INFO] Train: {len(train_dataset)} samples")
        print(f"[INFO] Val  : {len(val_dataset)} samples")

    return train_loader, val_loader