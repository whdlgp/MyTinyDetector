from pathlib import Path
import yaml
import torch
from torchvision.ops import box_iou

from block.model import Detector, decode
from dataloader.pascalvoc_trainval import get_dataloader, VOC_CLASSES


def match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thresh):
    # pred_boxes: (N, 4), gt_boxes: (M, 4) -> tp, fp counts and number of matched gt
    tp = 0
    fp = 0
    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), matched_gt

    ious = box_iou(pred_boxes, gt_boxes)  # (N, M)

    for i in range(len(pred_boxes)):
        candidate_mask = (gt_cls == pred_cls[i]) & (~matched_gt)
        if candidate_mask.sum() == 0:
            fp += 1
            continue

        candidate_ious = ious[i].clone()
        candidate_ious[~candidate_mask] = -1
        best_iou, best_idx = candidate_ious.max(dim=0)

        if best_iou >= iou_thresh:
            tp += 1
            matched_gt[best_idx] = True
        else:
            fp += 1

    return tp, fp, matched_gt


def evaluate(model, loader, device, conf_thresh, nms_thresh, iou_thresh=0.5):
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for images, boxes, labels in loader:
            images = images.to(device)
            output = model(images)
            results = decode(output, conf_thresh=conf_thresh, nms_thresh=nms_thresh)

            for i in range(len(results)):
                pred_boxes = results[i]["boxes"].cpu()
                pred_cls = results[i]["class_id"].cpu()

                gt_boxes = torch.tensor(boxes[i]) if len(boxes[i]) > 0 else torch.zeros(0, 4)
                gt_cls = torch.tensor(labels[i]) if len(labels[i]) > 0 else torch.zeros(0, dtype=torch.long)

                tp, fp, matched_gt = match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thresh)
                fn = len(gt_boxes) - matched_gt.sum().item()

                total_tp += tp
                total_fp += fp
                total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return precision, recall, total_tp, total_fp, total_fn


def draw(results, save_path="pr_result.png"):
    import matplotlib.pyplot as plt

    conf_thresh = [r["conf_thresh"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    f1 = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(precision, recall)]
    best_idx = f1.index(max(f1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(conf_thresh, precision, marker="o", label="Precision")
    ax.plot(conf_thresh, recall, marker="o", label="Recall")
    ax.axvline(conf_thresh[best_idx], color="red", alpha=0.3, linestyle=":",
               label=f"best F1={f1[best_idx]:.3f} @ conf={conf_thresh[best_idx]:.2f}")

    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150)
    print(f"best F1: {f1[best_idx]:.4f} at conf_thresh={conf_thresh[best_idx]:.2f}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Detector(num_classes=len(VOC_CLASSES)).to(device)
    ckpt = torch.load(Path(config["checkpoint_dir"]) / config["test_checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_loader = get_dataloader(root=config["data_root"], image_set="val", img_size=config["img_size"], batch_size=8, shuffle=False)

    results = []

    for conf_thresh in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        precision, recall, tp, fp, fn = evaluate(model, val_loader, device, conf_thresh=conf_thresh, nms_thresh=config["nms_th"], iou_thresh=0.5)

        results.append({
            "conf_thresh": conf_thresh,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall
        })

    with open("evaluation_result.csv", "w") as f:
        f.write("conf_thresh,tp,fp,fn,precision,recall\n")
        for result in results:
            f.write(f"{result["conf_thresh"]:.2f},{result["tp"]},{result["fp"]},{result["fn"]},{result["precision"]:.4f},{result["recall"]:.4f}\n")

    draw(results)