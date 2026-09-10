import torch
import torch.nn as nn
import torchvision

from .backbone import ConvNeXtBackbone
from .head import SimpleHead


class Detector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ConvNeXtBackbone()
        self.head = SimpleHead(in_channels=self.backbone.out_channels, num_classes=num_classes)
        self.stride = self.backbone.stride * self.head.stride # 32 x 2 = 64

    def forward(self, x):
        feat = self.backbone(x)
        
        # (B, 5+num_classes, H/64, W/64), raw output
        return self.head(feat)


def decode(output, conf_thresh=0.5, nms_thresh=0.5):
    B, C, H, W = output.shape
    output = output.permute(0, 2, 3, 1)

    obj = torch.sigmoid(output[..., 0])
    cy, cx = torch.meshgrid(torch.arange(H, device=output.device), torch.arange(W, device=output.device), indexing="ij")
    bx = (torch.sigmoid(output[..., 1]) + cx) / W
    by = (torch.sigmoid(output[..., 2]) + cy) / H
    bw = torch.sigmoid(output[..., 3])
    bh = torch.sigmoid(output[..., 4])
    cls_probs = torch.softmax(output[..., 5:], dim=-1)
    scores, cls_id = cls_probs.max(dim=-1)
    combined_score = obj * scores
    boxes = torch.stack([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2], dim=-1)

    results = []
    for b in range(B):
        mask = combined_score[b] > conf_thresh
        b_boxes = boxes[b][mask]
        b_obj = obj[b][mask]
        b_cls_id = cls_id[b][mask]
        b_scores = scores[b][mask]

        final_scores = b_obj * b_scores
        keep = torchvision.ops.nms(b_boxes, final_scores, nms_thresh)

        results.append({
            "boxes": b_boxes[keep],
            "objectness": b_obj[keep],
            "class_id": b_cls_id[keep],
            "class_score": b_scores[keep],
        })
    return results