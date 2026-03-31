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
# 2. ChunkedDiskDataset  –  reads one chunk-file at a time
# ─────────────────────────────────────────────
class ChunkedDiskDataset(Dataset):
    """
    Reads pre-processed samples that were saved as small chunk files
    (chunk_0000.pt, chunk_0001.pt, …) instead of one giant .pt file.

    Only one chunk is kept in RAM at a time.

    Directory layout expected:
        save_pt_dir/
            split_imgsize/
                index.pt          ← list of (chunk_file, local_idx) tuples
                chunk_0000.pt     ← list of sample dicts  (CHUNK_SIZE items)
                chunk_0001.pt
                …
    """

    CHUNK_SIZE = 200   # samples per chunk file  ← tune to your RAM budget

    def __init__(self, chunk_dir: str):
        self.chunk_dir  = chunk_dir
        self.index      = torch.load(os.path.join(chunk_dir, "index.pt"),
                                     weights_only=False)
        # cache: (chunk_id, list-of-samples) — replaced when chunk_id changes
        self._cache_id  = None
        self._cache     = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        chunk_file, local_idx = self.index[idx]

        if self.chunk_dir != self._cache_id or self._cache is None:
            # load only when we switch to a different chunk
            pass

        chunk_id = chunk_file
        if chunk_id != self._cache_id:
            self._cache    = torch.load(os.path.join(self.chunk_dir, chunk_file),
                                        weights_only=False)
            self._cache_id = chunk_id

        return self._cache[local_idx]

    # ── class method: build chunks from a SBInstanceDataset ──────────────
    @classmethod
    def build(cls, src_dataset: SBInstanceDataset,
              chunk_dir: str, verbose: bool = True) -> "ChunkedDiskDataset":
        """
        Iterate src_dataset one sample at a time, accumulate CHUNK_SIZE
        samples, save to disk, then discard from RAM.
        """
        os.makedirs(chunk_dir, exist_ok=True)

        index        = []          # (chunk_filename, local_idx_within_chunk)
        chunk_buf    = []
        chunk_id     = 0
        total        = len(src_dataset)

        iterator = tqdm(range(total), desc=f"Building {chunk_dir}") if verbose \
                   else range(total)

        for global_idx in iterator:
            sample = src_dataset[global_idx]
            local_idx = len(chunk_buf)
            chunk_filename = f"chunk_{chunk_id:04d}.pt"
            index.append((chunk_filename, local_idx))
            chunk_buf.append(sample)

            if len(chunk_buf) >= cls.CHUNK_SIZE:
                torch.save(chunk_buf,
                           os.path.join(chunk_dir, chunk_filename))
                chunk_buf  = []        # ← release RAM
                chunk_id  += 1

        # flush final partial chunk
        if chunk_buf:
            chunk_filename = f"chunk_{chunk_id:04d}.pt"
            # patch last entries in index that point to this file
            for i in range(len(chunk_buf)):
                index[-(len(chunk_buf) - i)] = (chunk_filename, i)
            torch.save(chunk_buf, os.path.join(chunk_dir, chunk_filename))

        torch.save(index, os.path.join(chunk_dir, "index.pt"))

        if verbose:
            print(f"[INFO] Saved {total} samples in "
                  f"{chunk_id + 1} chunks → {chunk_dir}")

        return cls(chunk_dir)


# ─────────────────────────────────────────────
# 3. Online-augmentation wrapper
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
# 4. collate_fn
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
# 5. Public API
# ─────────────────────────────────────────────
def get_sbd_dataloaders(
    root: str   = "/data/SBD",
    batch_size  : int  = 4,
    num_workers : int  = 2,
    img_size    : int  = 550,
    verbose     : bool = True,
    save_pt_dir : str  = "processed_data",
    seed        : int  = 42,
):
    """
    Build (or load from disk) SBD dataloaders.

    Chunk files are stored under:
        save_pt_dir/train_{img_size}/   ← ChunkedDiskDataset
        save_pt_dir/val_{img_size}/

    RAM usage during build ≈  CHUNK_SIZE × (one image + masks).
    RAM usage during training ≈  one chunk loaded per split.
    """
    os.makedirs(save_pt_dir, exist_ok=True)
    os.makedirs(root,        exist_ok=True)

    # ── locate / download / extract raw data ──────────────────────────────
    dataset_path = os.path.join(root, "dataset")
    os.makedirs(dataset_path, exist_ok=True)

    # promote files that ended up directly in root
    for item in ["img", "cls", "inst", "train.txt", "val.txt"]:
        src = os.path.join(root, item)
        dst = os.path.join(dataset_path, item)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)

    if not os.path.exists(os.path.join(dataset_path, "train.txt")):
        url      = ("https://www2.eecs.berkeley.edu/Research/Projects/CS/"
                    "vision/grouping/semantic_contours/benchmark.tgz")
        tgz_path = os.path.join(root, "benchmark.tgz")
        if not os.path.exists(tgz_path):
            if verbose:
                print("[INFO] Downloading SBD dataset…")
            urllib.request.urlretrieve(url, tgz_path)

        if verbose:
            print("[INFO] Extracting SBD dataset…")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(root)

        src_release = os.path.join(root, "benchmark_RELEASE", "dataset")
        for item in os.listdir(src_release):
            target = os.path.join(dataset_path, item)
            if not os.path.exists(target):
                shutil.move(os.path.join(src_release, item), target)
        shutil.rmtree(os.path.join(root, "benchmark_RELEASE"),
                      ignore_errors=True)

    # hide .tgz so torchvision doesn't try to re-extract it
    tgz_path = os.path.join(root, "benchmark.tgz")
    tgz_bak  = tgz_path + ".bak"
    if os.path.exists(tgz_path):
        if os.path.exists(tgz_bak):
            os.remove(tgz_bak)
        os.rename(tgz_path, tgz_bak)

    try:
        base_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # ── train ──────────────────────────────────────────────────────────
        train_chunk_dir = os.path.join(save_pt_dir, f"train_{img_size}")
        if os.path.exists(os.path.join(train_chunk_dir, "index.pt")):
            if verbose:
                print(f"[INFO] Loading cached train chunks from {train_chunk_dir}")
            train_base = ChunkedDiskDataset(train_chunk_dir)
        else:
            if verbose:
                print(f"[INFO] Building train chunks → {train_chunk_dir}")
            ds_train   = SBInstanceDataset(data_root=dataset_path,
                                           image_set="train",
                                           transforms=base_transform)
            train_base = ChunkedDiskDataset.build(ds_train, train_chunk_dir,
                                                  verbose=verbose)

        # ── val ────────────────────────────────────────────────────────────
        val_chunk_dir = os.path.join(save_pt_dir, f"val_{img_size}")
        if os.path.exists(os.path.join(val_chunk_dir, "index.pt")):
            if verbose:
                print(f"[INFO] Loading cached val chunks from {val_chunk_dir}")
            val_base = ChunkedDiskDataset(val_chunk_dir)
        else:
            if verbose:
                print(f"[INFO] Building val chunks → {val_chunk_dir}")
            ds_val   = SBInstanceDataset(data_root=dataset_path,
                                         image_set="val",
                                         transforms=base_transform)
            val_base = ChunkedDiskDataset.build(ds_val, val_chunk_dir,
                                                verbose=verbose)

    finally:
        if os.path.exists(tgz_bak):
            os.rename(tgz_bak, tgz_path)

    # ── online augmentation only for train ────────────────────────────────
    online_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, p=0.5),
    ])

    train_loader = DataLoader(
        AugmentedDataset(train_base, online_aug, img_size),
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_base,
        batch_size  = 1,
        shuffle     = False,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )

    if verbose:
        print(f"[INFO] Train: {len(train_base)} samples | "
              f"Val: {len(val_base)} samples")

    return train_loader, val_loader