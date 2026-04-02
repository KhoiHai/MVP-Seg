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
    def __init__(self, data_root: str, image_set: str = "train", transforms=None):
        self.dataset    = SBDataset(root=data_root, image_set=image_set,
                                    mode="segmentation", download=False)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, mask = self.dataset[idx]
        image = np.array(image)
        mask  = np.array(mask)

        class_ids = np.unique(mask)
        class_ids = class_ids[(class_ids != 0) & (class_ids != 255)]

        masks_list, labels_list = [], []
        for cls_id in class_ids:
            binary        = (mask == cls_id).astype(np.uint8)
            labeled_array, num_instances = ndimage.label(binary)
            for inst_idx in range(1, num_instances + 1):
                inst_mask = (labeled_array == inst_idx).astype(np.uint8)
                if inst_mask.sum() < 20:
                    continue
                masks_list.append(inst_mask)
                labels_list.append(int(cls_id) - 1)

        if self.transforms and len(masks_list) > 0:
            transformed = self.transforms(image=image, masks=masks_list)
            image       = transformed["image"]
            masks_list  = transformed["masks"]
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
                "boxes":  torch.zeros((0, 4),    dtype=torch.float32),
                "labels": torch.zeros((0,),      dtype=torch.int64),
                "masks":  torch.zeros((0, h, w), dtype=torch.float32),
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
    Directory layout:
        chunk_dir/
            index.pt          ← list of (chunk_filename, local_idx) tuples
            chunk_0000.pt     ← list of sample dicts
            chunk_0001.pt
            …
    """

    CHUNK_SIZE = 200

    def __init__(self, chunk_dir: str):
        self.chunk_dir = chunk_dir
        self.index     = torch.load(os.path.join(chunk_dir, "index.pt"),
                                    weights_only=False)
        self._cache_id = None
        self._cache    = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        chunk_file, local_idx = self.index[idx]
        if chunk_file != self._cache_id:
            self._cache    = torch.load(
                os.path.join(self.chunk_dir, chunk_file),
                weights_only=False
            )
            self._cache_id = chunk_file
        return self._cache[local_idx]

    @classmethod
    def build(cls, src_dataset: SBInstanceDataset,
              chunk_dir: str, verbose: bool = True) -> "ChunkedDiskDataset":
        os.makedirs(chunk_dir, exist_ok=True)

        index     = []
        chunk_buf = []
        chunk_id  = 0
        total     = len(src_dataset)

        iterator = tqdm(range(total), desc=f"Building {chunk_dir}") \
                   if verbose else range(total)

        for global_idx in iterator:
            sample         = src_dataset[global_idx]
            local_idx      = len(chunk_buf)
            chunk_filename = f"chunk_{chunk_id:04d}.pt"
            index.append((chunk_filename, local_idx))
            chunk_buf.append(sample)

            if len(chunk_buf) >= cls.CHUNK_SIZE:
                torch.save(chunk_buf,
                           os.path.join(chunk_dir, chunk_filename))
                chunk_buf = []
                chunk_id += 1

        # flush chunk cuối
        if chunk_buf:
            chunk_filename = f"chunk_{chunk_id:04d}.pt"
            n = len(chunk_buf)
            for i in range(n):
                index[-(n - i)] = (chunk_filename, i)
            torch.save(chunk_buf, os.path.join(chunk_dir, chunk_filename))

        torch.save(index, os.path.join(chunk_dir, "index.pt"))

        if verbose:
            print(f"[INFO] Saved {total} samples in "
                  f"{chunk_id + 1} chunks → {chunk_dir}")

        return cls(chunk_dir)


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
    root        : str  = "/data/SBD",
    batch_size  : int  = 4,
    num_workers : int  = 2,
    img_size    : int  = 550,
    verbose     : bool = True,
    save_pt_dir : str  = "processed_data",
):
    train_chunk_dir = os.path.join(save_pt_dir, f"train_{img_size}")
    val_chunk_dir   = os.path.join(save_pt_dir, f"val_{img_size}")

    train_cached = os.path.exists(os.path.join(train_chunk_dir, "index.pt"))
    val_cached   = os.path.exists(os.path.join(val_chunk_dir,   "index.pt"))

    # ✅ Cả 2 split đã có sẵn → load thẳng, skip toàn bộ raw data
    if train_cached and val_cached:
        if verbose:
            print(f"[INFO] Loading cached train chunks from {train_chunk_dir}")
            print(f"[INFO] Loading cached val   chunks from {val_chunk_dir}")
        train_dataset = ChunkedDiskDataset(train_chunk_dir)
        val_dataset   = ChunkedDiskDataset(val_chunk_dir)

    else:
        # ── Cần build chunk → phải có raw data ───────────────────────────
        os.makedirs(save_pt_dir, exist_ok=True)
        os.makedirs(root,        exist_ok=True)

        dataset_path = os.path.join(root, "dataset")
        os.makedirs(dataset_path, exist_ok=True)

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

        # hide .tgz để torchvision không re-extract
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

            # ── train ──────────────────────────────────────────────────────
            if train_cached:
                if verbose:
                    print(f"[INFO] Loading cached train chunks from {train_chunk_dir}")
                train_dataset = ChunkedDiskDataset(train_chunk_dir)
            else:
                if verbose:
                    print(f"[INFO] Building train chunks → {train_chunk_dir}")
                ds_train      = SBInstanceDataset(data_root=dataset_path,
                                                  image_set="train",
                                                  transforms=base_transform)
                train_dataset = ChunkedDiskDataset.build(ds_train, train_chunk_dir,
                                                         verbose=verbose)

            # ── val ────────────────────────────────────────────────────────
            if val_cached:
                if verbose:
                    print(f"[INFO] Loading cached val chunks from {val_chunk_dir}")
                val_dataset = ChunkedDiskDataset(val_chunk_dir)
            else:
                if verbose:
                    print(f"[INFO] Building val chunks → {val_chunk_dir}")
                ds_val      = SBInstanceDataset(data_root=dataset_path,
                                                image_set="val",
                                                transforms=base_transform)
                val_dataset = ChunkedDiskDataset.build(ds_val, val_chunk_dir,
                                                       verbose=verbose)
        finally:
            if os.path.exists(tgz_bak):
                os.rename(tgz_bak, tgz_path)

    # ── dataloaders ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )

    if verbose:
        print(f"[INFO] Train: {len(train_dataset)} samples | "
              f"Val: {len(val_dataset)} samples")

    return train_loader, val_loader