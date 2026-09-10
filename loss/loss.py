import torch
import torch.nn.functional as F
import torchvision


def build_target(boxes_batch, labels_batch, grid_size, num_classes):
    # boxes_batch: list of (N_i, 4)
    # labels_batch: list of (N_i,) 

    B = len(boxes_batch)
    H = W = grid_size
    target = torch.zeros(B, 5 + num_classes, H, W)

    for b in range(B):
        for box, label in zip(boxes_batch[b], labels_batch[b]):
            xmin, ymin, xmax, ymax = box
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            w, h = xmax - xmin, ymax - ymin

            col = min(int(cx * W), W - 1)
            row = min(int(cy * H), H - 1)

            target[b, 0, row, col] = 1.0
            target[b, 1, row, col] = cx * W - col  # tx
            target[b, 2, row, col] = cy * H - row  # ty
            target[b, 3, row, col] = w
            target[b, 4, row, col] = h
            target[b, 5 + int(label), row, col] = 1.0

    # target: (B, 5+num_classes, H, W)
    return target


def compute_loss(output, target, lambda_coord=3.0, alpha_obj=0.25, gamma_obj=2.0):
    # output: (B, 5+num_classes, H, W)
    # target: (B, 5+num_classes, H, W)
    # lambda_coord: box_loss multiplier
    # alpha_obj: objectness positive sample weight
    # gamma_obj: objectness focusing parameter
    obj_mask = target[:, 0] > 0  # (B, H, W)

    raw_obj = output[:, 0]  # (B, H, W), raw logit before sigmoid
    target_obj = target[:, 0]  # (B, H, W)
    obj_loss = torchvision.ops.sigmoid_focal_loss(
        raw_obj, target_obj, alpha=alpha_obj, gamma=gamma_obj, reduction="mean"
    )

    pred_box = torch.sigmoid(output[:, 1:5]).permute(0, 2, 3, 1)[obj_mask]  # (N, 4)
    target_box = target[:, 1:5].permute(0, 2, 3, 1)[obj_mask]  # (N, 4)

    # (N, 4): tx, ty, w, h -> (N, 4): tx, ty, sqrt(w), sqrt(h)
    pred_box = torch.cat([pred_box[:, :2], torch.sqrt(pred_box[:, 2:].clamp(min=1e-6))], dim=-1)
    target_box = torch.cat([target_box[:, :2], torch.sqrt(target_box[:, 2:].clamp(min=1e-6))], dim=-1)

    pred_cls = torch.softmax(output[:, 5:], dim=1).permute(0, 2, 3, 1)[obj_mask]  # (N, num_classes)
    target_cls = target[:, 5:].permute(0, 2, 3, 1)[obj_mask]  # (N, num_classes)

    if pred_box.numel() == 0:
        zero = torch.tensor(0.0, device=output.device)
        return {
            "total_loss": obj_loss,
            "obj_loss": obj_loss,
            "box_loss": zero,
            "cls_loss": zero,
            # Only for debugging
            "tx_loss": zero,
            "ty_loss": zero,
            "w_loss": zero,
            "h_loss": zero,
        }

    box_loss = F.mse_loss(pred_box, target_box)
    cls_loss = F.mse_loss(pred_cls, target_cls)

    total_loss = obj_loss + cls_loss + lambda_coord * box_loss
    
    # Only for debugging
    tx_loss = F.mse_loss(pred_box[:, 0], target_box[:, 0])
    ty_loss = F.mse_loss(pred_box[:, 1], target_box[:, 1])
    w_loss = F.mse_loss(pred_box[:, 2], target_box[:, 2])
    h_loss = F.mse_loss(pred_box[:, 3], target_box[:, 3])

    return {
        "total_loss": total_loss,
        "obj_loss": obj_loss,
        "box_loss": box_loss,
        "cls_loss": cls_loss,
        # Only for debugging
        "tx_loss": tx_loss,
        "ty_loss": ty_loss,
        "w_loss": w_loss,
        "h_loss": h_loss,
    }