import torch
from .loss import cls_loss, mask_loss, ciou_loss
from src.utils.generate_masks import generate_masks
from src.utils.concat_prediction import concat_prediction

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_step(self, images, gt_boxes, cls_target, mask_target):
        """
        images: [B,3,H,W]
        gt_boxes: [B,N,4] x1,y1,x2,y2
        cls_target: [B,N]
        mask_target: [B,N,Hm,Wm] resized to proto size
        """
        self.model.train()
        images = images.to(self.device)
        gt_boxes = gt_boxes.to(self.device)
        cls_target = cls_target.to(self.device)
        mask_target = mask_target.to(self.device)

        outputs = self.model(images)

        # Flatten multi-scale outputs
        cls_pred = concat_prediction(outputs['cls_out'])
        box_pred = concat_prediction(outputs['box_out'])
        coef_pred = concat_prediction(outputs['coef_out'])
        proto = outputs['proto_out']

        # Generate mask prediction
        mask_pred = generate_masks(proto, coef_pred)

        # Compute loss
        loss_cls = cls_loss(cls_pred, cls_target)
        loss_box = ciou_loss(box_pred.view(-1,4), gt_boxes.view(-1,4))
        loss_mask = mask_loss(mask_pred, mask_target)

        total_loss = loss_cls + loss_box + loss_mask

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()