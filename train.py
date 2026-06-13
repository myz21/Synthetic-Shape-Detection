"""
train.py
────────
Training pipeline.

Required components (§7):
    - PyTorch Dataset
    - PyTorch DataLoader
    - Loss function
    - Optimizer
    - Learning rate
    - Fixed number of epochs or stopping criterion
    - Validation-set monitoring
    - Saved quantitative results
    - Prediction visualizations

Training details to report (§7):
    input image size, batch size, number of epochs, optimizer,
    learning rate, loss function, pretrained weights (yes/no),
    hardware, approximate training time
"""

import torch
from torch.utils.data import DataLoader

from src.dataset import SyntheticShapeDataset, build_coco_bases, get_split_ids


def get_transforms(split: str):
    """
    Return torchvision transforms for the given split.
    At minimum: Resize, ToTensor, Normalize.
    """
    raise NotImplementedError


def build_model(task: str, pretrained: bool):
    """
    Build and return a CNN-based detection or segmentation model.

    Detection options (§6):
        Faster R-CNN, SSD-style, YOLO-style, custom CNN detector

    Segmentation options (§6):
        U-Net, FCN-style, encoder-decoder CNN, custom CNN

    Clearly state which parts are pretrained and which are trained from scratch.
    """
    raise NotImplementedError


def loss_fn(predictions, targets):
    """
    Define and return the loss value for one batch.
    Choose a loss appropriate for the task (§7).
    """
    raise NotImplementedError


def train_one_epoch(model, loader, optimizer, device) -> float:
    """Run one training epoch; return average loss."""
    raise NotImplementedError


def evaluate(model, loader, device) -> dict:
    """
    Evaluate on val set; return metric dict.
    Do NOT call this on the test set for model selection (§7).
    """
    raise NotImplementedError


def save_results(metrics: dict, epoch: int):
    """Save per-epoch metrics to results/metrics.json."""
    import json, time
    from pathlib import Path

    out = Path("results") / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    history = json.loads(out.read_text()) if out.exists() else []
    history.append({"epoch": epoch, "timestamp": time.time(), **metrics})
    out.write_text(json.dumps(history, indent=2))


def main():
    TASK      = "detection"        # or "segmentation"
    PRETRAINED = True
    DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

    train_base, val_base = build_coco_bases("data/coco")

    # Fixed split — given by the assignment
    train_ids = get_split_ids(train_base, "train")   # first 5000
    val_ids   = get_split_ids(val_base,   "val")     # first 1000

    train_ds = SyntheticShapeDataset(train_base, train_ids, "train", TASK, get_transforms("train"))
    val_ds   = SyntheticShapeDataset(val_base,   val_ids,   "val",   TASK, get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=8,  shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=8,  shuffle=False, num_workers=4)

    model     = build_model(TASK, PRETRAINED).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, 11):
        train_loss  = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch}  loss={train_loss:.4f}  metrics={val_metrics}")
        save_results({"train_loss": train_loss, **val_metrics}, epoch)


if __name__ == "__main__":
    main()
