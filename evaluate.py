"""
evaluate.py
───────────
Evaluation on the fixed test set.

Required metrics — Detection (§9):
    - Precision at IoU threshold 0.5
    - Recall    at IoU threshold 0.5
    - F1-score  at IoU threshold 0.5
    - Mean IoU  of matched predictions

Required metrics — Segmentation (§9):
    - Foreground IoU / Jaccard score
    - Dice coefficient
    - Foreground precision
    - Foreground recall
    (pixel accuracy alone is NOT sufficient)

Required visualizations (§9):
    At least 12 test images including successful predictions,
    failure cases, positive images, and negative images.

Baseline comparison (§8):
    Compare CNN model against at least one simple baseline:
    color thresholding, edge detection + connected components,
    template matching, logistic regression, or shallow CNN.
"""

import torch
from torch.utils.data import DataLoader

from src.dataset import SyntheticShapeDataset, build_coco_bases, get_split_ids
from train import build_model, get_transforms


def detection_metrics(predictions: list, targets: list, iou_threshold: float = 0.5) -> dict:
    """
    Compute Precision, Recall, F1 at IoU=0.5 and mean IoU (§9).
    Match predicted boxes to ground-truth boxes via greedy IoU matching.
    """
    raise NotImplementedError


def segmentation_metrics(predictions: list, targets: list) -> dict:
    """
    Compute foreground IoU, Dice, Precision, Recall (§9).
    Do NOT report pixel accuracy as the sole metric.
    """
    raise NotImplementedError


def visualize_predictions(images, targets, predictions, n: int = 12, save_dir: str = "figures"):
    """
    Save visualizations for at least 12 test images (§9).
    Must include: successful predictions, failure cases,
    positive images, and negative images.
    """
    raise NotImplementedError


class Baseline:
    """
    Simple baseline to compare against the CNN model (§8).
    Choose one: color thresholding, edge detection + connected components,
    template matching, logistic regression, or shallow CNN.
    """

    def predict(self, image):
        raise NotImplementedError


def main():
    TASK   = "detection"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_base = build_coco_bases("data/coco")

    # Fixed test split — given by the assignment (do NOT use for model selection)
    test_ids = get_split_ids(val_base, "test")   # val2017 indices 1000–1999

    test_ds     = SyntheticShapeDataset(val_base, test_ids, "test", TASK, get_transforms("val"))
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

    model = build_model(TASK, pretrained=False).to(DEVICE)
    # model.load_state_dict(torch.load("results/best_model.pth", map_location=DEVICE))
    model.eval()

    all_predictions, all_targets, all_images = [], [], []

    with torch.no_grad():
        for images, targets in test_loader:
            pass  # collect predictions

    if TASK == "detection":
        metrics = detection_metrics(all_predictions, all_targets)
    else:
        metrics = segmentation_metrics(all_predictions, all_targets)

    print("Test metrics:", metrics)

    visualize_predictions(all_images, all_targets, all_predictions, n=12)


if __name__ == "__main__":
    main()
