import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from timm.data.transforms_factory import create_transform
from torchvision.ops import nms

from src.models.mvp_seg import MVPSeg


# 80 COCO class names (index 0-based)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# One fixed color per class for consistent visualization
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)


def flatten_predictions(feature_list):
    '''
    Flatten multi-scale prediction tensors into a single sequence
        Converts list of [B, C, Hi, Wi] -> [B, sum(Hi*Wi), C]
    Args:
        feature_list: list of Tensors [B, C, Hi, Wi]
    Returns:
        Tensor [B, total_anchors, C]
    '''
    out = []
    for f in feature_list:
        B, C, H, W = f.shape
        out.append(f.permute(0, 2, 3, 1).reshape(B, -1, C))
    return torch.cat(out, dim=1)


def decode_single_image(outputs, score_thresh=0.3, nms_thresh=0.5,
                         top_k=200, mask_thresh=0.5, img_size=224):
    '''
    Decode raw model outputs of the FIRST image in the batch into detections
        Steps: flatten -> softmax -> top-k -> NMS -> mask reconstruction -> upsample
        Note: Works on untrained models too (random weights -> random predictions)
    Args:
        outputs     : dict from MVPSeg.forward()
        score_thresh: Minimum confidence score to keep a detection
        nms_thresh  : IoU threshold for NMS
        top_k       : Max candidates before NMS
        mask_thresh : Threshold to binarize sigmoid mask
        img_size    : Original image size to upsample masks back to
    Returns:
        boxes  : np.ndarray [K, 4]  x1,y1,x2,y2 (scaled to img_size)
        scores : np.ndarray [K]
        labels : np.ndarray [K]     0-based class indices
        masks  : np.ndarray [K, img_size, img_size]  binary uint8
    '''
    cls_preds  = flatten_predictions(outputs["cls"])   # [B, #pred, num_classes]
    box_preds  = flatten_predictions(outputs["box"])   # [B, #pred, 4]
    coef_preds = flatten_predictions(outputs["coef"])  # [B, #pred, P]
    proto      = outputs["proto"]                      # [B, P, H/4, W/4]

    # Work on first image only
    cls_p  = cls_preds[0]    # [#pred, num_classes]
    box_p  = box_preds[0]    # [#pred, 4]
    coef_p = coef_preds[0]   # [#pred, P]
    proto_i = proto[0]       # [P, H/4, W/4]

    # Softmax -> score and label per anchor
    scores, labels = cls_p.softmax(-1).max(-1)  # [#pred]

    # Top-k selection
    k = min(top_k, scores.shape[0])
    topk_scores, topk_idx = scores.topk(k)
    box_p  = box_p[topk_idx]
    labels = labels[topk_idx]
    coef_p = coef_p[topk_idx]

    # Score threshold filter
    keep   = topk_scores >= score_thresh
    if keep.sum() == 0:
        # Untrained model likely returns all low scores -> relax threshold automatically
        print(f"[Info] No detections above score_thresh={score_thresh}. "
              f"Showing top-3 predictions regardless (model not trained).")
        _, keep_idx = topk_scores.topk(min(3, k))
        keep = torch.zeros(k, dtype=torch.bool)
        keep[keep_idx] = True

    box_p  = box_p[keep]
    labels = labels[keep]
    coef_p = coef_p[keep]
    scores = topk_scores[keep]

    # NMS
    if box_p.shape[0] > 1:
        keep_nms = nms(box_p, scores, nms_thresh)
        box_p    = box_p[keep_nms]
        labels   = labels[keep_nms]
        coef_p   = coef_p[keep_nms]
        scores   = scores[keep_nms]

    # Reconstruct masks: M = sigmoid(proto @ coef^T)
    Ph, Pw     = proto_i.shape[1], proto_i.shape[2]
    proto_flat = proto_i.permute(1, 2, 0).reshape(-1, proto_i.shape[0])
    # [H/4*W/4, P] @ [P, K] -> [H/4*W/4, K]
    pred_masks = torch.sigmoid(proto_flat @ coef_p.T)
    pred_masks = pred_masks.reshape(Ph, Pw, -1).permute(2, 0, 1)        # [K, H/4, W/4]

    # Upsample masks to original image size
    pred_masks = F.interpolate(
        pred_masks.unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False
    ).squeeze(0)                                                          # [K, img_size, img_size]
    pred_masks = (pred_masks > mask_thresh).cpu().numpy().astype(np.uint8)

    # Scale boxes from feature-map coordinates to image pixel coordinates
    # box_preds are raw (non-normalized) -> clamp to [0, img_size]
    box_np = box_p.cpu().numpy()
    box_np = np.clip(box_np, 0, img_size)

    return box_np, scores.cpu().numpy(), labels.cpu().numpy(), pred_masks


def visualize(img_path, model_name="nvidia/MambaVision-T-1K",
              score_thresh=0.3, nms_thresh=0.5, top_k=200, mask_thresh=0.5):
    '''
    Run MVP-Seg on a single image and visualize bounding boxes + segmentation masks
        Works with untrained weights (predictions will be random but pipeline is verified)
        Each detected object gets a colored mask overlay and a labeled bounding box
    Args:
        img_path    : Path to input image
        model_name  : HuggingFace MambaVision model name
        score_thresh: Confidence threshold for detections
        nms_thresh  : NMS IoU threshold
        top_k       : Max candidates before NMS
        mask_thresh : Sigmoid threshold for mask binarization
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (pretrained backbone, random head weights if not fine-tuned)
    model = MVPSeg(model_name=model_name, num_classes=80, num_prototypes=32).to(device)
    model.eval()

    # Load and preprocess image using backbone's own normalization config
    image_pil = Image.open(img_path).convert("RGB")
    img_size  = 224

    transform = create_transform(
        input_size  = (3, img_size, img_size),
        is_training = False,
        mean        = model.backbone.model.config.mean,
        std         = model.backbone.model.config.std,
        crop_mode   = model.backbone.model.config.crop_mode,
        crop_pct    = model.backbone.model.config.crop_pct,
    )

    x = transform(image_pil).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(x)

    # Decode predictions
    boxes, scores, labels, masks = decode_single_image(
        outputs,
        score_thresh = score_thresh,
        nms_thresh   = nms_thresh,
        top_k        = top_k,
        mask_thresh  = mask_thresh,
        img_size     = img_size
    )

    # Prepare display image (resize original PIL to match model input size)
    img_display = np.array(image_pil.resize((img_size, img_size)))

    # Build mask overlay: blend each mask with its class color
    overlay = img_display.copy().astype(np.float32)
    for j in range(len(boxes)):
        color = COLORS[int(labels[j]) % 80].astype(np.float32)
        mask  = masks[j]                                            # [H, W] binary
        overlay[mask == 1] = overlay[mask == 1] * 0.5 + color * 0.5

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Plot side by side: original | masked overlay with boxes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(img_display)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"MVP-Seg Output ({len(boxes)} detections)", fontsize=13)
    axes[1].axis("off")

    # Draw bounding boxes and labels on the right panel
    for j in range(len(boxes)):
        x1, y1, x2, y2 = boxes[j]
        color_f = tuple(COLORS[int(labels[j]) % 80] / 255.0)       # matplotlib float color

        # Rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth = 2, edgecolor = color_f, facecolor = "none"
        )
        axes[1].add_patch(rect)

        # Label with class name and score
        class_name = COCO_CLASSES[int(labels[j]) % 80]
        label_text = f"{class_name} {scores[j]:.2f}"
        axes[1].text(
            x1, max(y1 - 4, 0), label_text,
            fontsize   = 8,
            color      = "white",
            bbox       = dict(facecolor=color_f, alpha=0.7, pad=1, edgecolor="none")
        )

    plt.tight_layout()
    plt.savefig("output_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to: output_visualization.png")


# ------------------------------
# VISUALIZATION TESTING
# ------------------------------
if __name__ == "__main__":
    visualize(
        img_path     = "data/bear.jpeg",
        model_name   = "nvidia/MambaVision-T-1K",
        score_thresh = 0.3,    # Lower this if no boxes appear (untrained model)
        nms_thresh   = 0.5,
        top_k        = 200,
        mask_thresh  = 0.5
    )