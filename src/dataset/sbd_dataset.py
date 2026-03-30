import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import SBDataset
from torchvision import transforms as T
import numpy as np
from PIL import Image

class SBInstanceDataset(Dataset):
    """
    SBDataset loader for instance segmentation
    Converts SBD semantic masks to instance masks
    """
    def __init__(self, root, image_set="train", transforms=None):
        """
        Args:
            root: folder to download/extract SBD dataset
            image_set: "train" / "val" / "train_noval"
            transforms: torchvision or albumentations transforms
        """
        self.dataset = SBDataset(
            root=root,
            image_set=image_set,
            mode="segmentation",
            download=True
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]  # image: PIL.Image, mask: PIL.Image
        image = np.array(image)           # [H, W, 3]
        mask  = np.array(mask)            # [H, W]

        # Convert mask to instances
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 255]  # 255 = ignore

        boxes, labels, masks = [], [], []

        for inst_id in instance_ids:
            m = (mask == inst_id).astype(np.uint8)        # single instance mask
            ys, xs = np.where(m)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            boxes.append([x1, y1, x2, y2])
            labels.append(inst_id)  # in SBD, id = class_id
            masks.append(torch.tensor(m, dtype=torch.float32))

        if len(boxes) == 0:
            # fallback for empty annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.float32)
        else:
            boxes  = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks  = torch.stack(masks, dim=0)  # [N, H, W]

        # Transform image to tensor
        if self.transforms:
            transformed = self.transforms(image=image, masks=masks)
            image = transformed["image"]
            masks = torch.stack([torch.tensor(m, dtype=torch.float32) for m in transformed["masks"]])
        
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


def get_sbd_dataloaders(root="datasets/SBD", batch_size=4, num_workers=2, img_size=550):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Define transforms
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.2,0.2,0.2, p=0.5),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ], masks=True)

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ], masks=True)

    train_dataset = SBInstanceDataset(root=root, image_set="train_noval", transforms=train_transform)
    val_dataset   = SBInstanceDataset(root=root, image_set="val", transforms=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader