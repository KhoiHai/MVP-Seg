import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class COCODataset(Dataset):
    '''
    COCO Dataset for Instance Segmentation
        Loads images and annotations (boxes, labels, masks) from MS COCO format
        Supports subsetting for limited compute environments (e.g. Google Colab)
        Category IDs are remapped from COCO sparse (1-90) to dense 0-based indices
    '''
    def __init__(self, img_dir, ann_file, transforms = None, subset_size = None):
        '''
        Args:
            img_dir : Path to image folder (e.g. data/coco/train2017/)
            ann_file : Path to COCO annotation JSON file
            transforms : Albumentations transform pipeline (handles image + box + mask jointly)
            subset_size : If set, limits the dataset to the first N images (for COCO subset ~10k)
        '''
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.transforms = transforms

        # Keep only images that have at least 1 non-crowd annotation
        self.img_ids = sorted(self.coco.getImgIds())
        self.img_ids = [
            img_id for img_id in self.img_ids
            if len(self.coco.getAnnIds(imgIds = img_id, iscrowd = False)) > 0
        ]

        # Limit dataset size for subset training
        if subset_size is not None:
            self.img_ids = self.img_ids[:subset_size]

        # Remap COCO category IDs (sparse 1-90) to dense 0-based class indices
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        self.label_to_cat_id = {idx: cat_id for cat_id, idx in self.cat_id_to_label.items()}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # Load image as numpy RGB array for albumentations compatibility
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        # Load all non-crowd annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds = img_id, iscrowd = False)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks = [], [], []
        for ann in anns:
            # Convert COCO bbox format [x, y, w, h] -> [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            x2, y2 = x + w, y + h

            # Skip degenerate boxes
            if w < 1 or h < 1:
                continue

            boxes.append([x, y, x2, y2])
            labels.append(self.cat_id_to_label[ann["category_id"]])

            # Decode polygon or RLE segmentation to binary mask [H, W]
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        # Skip images with no valid annotations after filtering
        if len(boxes) == 0:
            return None

        boxes  = np.array(boxes,  dtype = np.float32)                    # [N, 4]
        labels = np.array(labels, dtype = np.int64)                      # [N]
        masks  = np.stack(masks, axis=0).astype(np.float32)              # [N, H, W]

        # --- transform ---
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes.tolist(),
                bbox_labels=labels.tolist(),
                masks=[m for m in masks]
            )

            if len(transformed["bboxes"]) == 0:
                return None

            image = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["bbox_labels"], dtype=torch.int64)
            if len(transformed["masks"]) == 0:
                return None
            masks = torch.from_numpy(np.stack(transformed["masks"]).astype(np.float32)).float()
        else:
            # Fallback: convert to tensor manually without augmentation
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes = torch.from_numpy(boxes).float()
            labels = torch.from_numpy(labels).long()
            masks = torch.from_numpy(masks).float()

        resized_h, resized_w = int(image.shape[1]), int(image.shape[2])

        return{
            "image":  image,    # [3, img_size, img_size]
            "boxes":  boxes,    # [N, 4]  pascal_voc format
            "labels": labels,   # [N]     0-based class indices
            "masks":  masks,    # [N, img_size, img_size]  binary
            "img_id": int(img_id),    # original COCO image ID for evaluation
            "orig_size": (orig_h, orig_w),
            "resized_size": (resized_h, resized_w),
        }


def get_transforms(train = True, img_size = 550):
    '''
    Build Albumentations transform pipeline for train or val split
        img_size = 550 follows YOLACT standard input resolution
        Albumentations handles bounding boxes and masks automatically
    Args:
        train   : If True, applies augmentation (flip, color jitter)
        img_size: Target image size after resize
    Returns:
        Albumentations Compose pipeline
    '''
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p = 0.5),                                   # Random horizontal flip
            A.ColorJitter(brightness = 0.2, contrast = 0.2,
                          saturation = 0.2, p = 0.5),                    # Color augmentation
            A.Normalize(mean = [0.485, 0.456, 0.406],
                        std  = [0.229, 0.224, 0.225]),                   # ImageNet normalization
            ToTensorV2()
        ], bbox_params = A.BboxParams(
            format = "pascal_voc",            # [x1, y1, x2, y2]
            label_fields = ["bbox_labels"],
            min_visibility = 0.3              # Drop boxes cropped more than 70%
        ))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean = [0.485, 0.456, 0.406],
                        std  = [0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params = A.BboxParams(
            format = "pascal_voc",
            label_fields = ["bbox_labels"]
        ))


def collate_fn(batch):
    '''
    Custom collate function for variable-length annotations per image
        Default PyTorch collate cannot stack tensors of different shapes
        (each image has a different number of objects)
    Args:
        batch: list of dicts returned by COCODataset.__getitem__
    Returns:
        images : Tensor [B, 3, H, W]
        targets: list of dicts, one per image (boxes, labels, masks, img_id)
    '''
    # Filter out None entries (images with no valid annotations)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    images  = torch.stack([b["image"] for b in batch])  # [B, 3, H, W]
    targets = [{
        "boxes":  b["boxes"],
        "labels": b["labels"],
        "masks":  b["masks"],
        "img_id": b["img_id"]
    } for b in batch]

    return images, targets


def get_coco_dataloaders(data_root, batch_size = 5, num_workers = 2,
                    subset_size = 10000, img_size = 550, val_subset_size = None):
    '''
    Build train and val DataLoaders for COCO subset training
        Expected data_root structure:
            data_root/
                train2017/
                val2017/
                annotations/
                    instances_train2017.json
                    instances_val2017.json
    Args:
        data_root   : Root path to COCO dataset
        batch_size  : Training batch size (5 per proposal)
        num_workers : Number of parallel data loading workers
        subset_size : Number of training images to use (10000 per proposal)
        img_size    : Input image resolution (550 per YOLACT standard)
    Returns:
        train_loader, val_loader
    '''
    train_dataset = COCODataset(
        img_dir      = os.path.join(data_root, "train2017"),
        ann_file     = os.path.join(data_root, "annotations/instances_train2017.json"),
        transforms   = get_transforms(train = True,  img_size = img_size),
        subset_size  = subset_size
    )
    val_dataset = COCODataset(
        img_dir      = os.path.join(data_root, "val2017"),
        ann_file     = os.path.join(data_root, "annotations/instances_val2017.json"),
        transforms   = get_transforms(train = False, img_size = img_size),
        subset_size  = val_subset_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True        # Faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = 1,          # Batch size 1 for accurate per-image mAP evaluation
        shuffle     = False,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True
    )

    return train_loader, val_loader
